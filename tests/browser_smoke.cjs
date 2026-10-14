// Optional: node tests/browser_smoke.cjs REPORT_HTML SCREENSHOT.png
const { chromium } = require(process.env.RUN_DIG_PLAYWRIGHT || 'playwright');
const { pathToFileURL } = require('node:url');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

async function checkGraphRoutes(graph) {
  const collisions = await graph.evaluate(svg => {
    const boxes = [...svg.querySelectorAll('.graph-node')].map(node => {
      const b = node.getBBox(), m = node.transform.baseVal.consolidate()?.matrix;
      return {id: node.dataset.id || node.textContent, x: b.x + (m?.e || 0), y: b.y + (m?.f || 0), width: b.width, height: b.height};
    });
    const failures = [];
    for (const path of svg.querySelectorAll('.graph-edge')) {
      const length = path.getTotalLength();
      if (!length || /NaN|undefined|Infinity/.test(path.getAttribute('d'))) { failures.push({edge: path.dataset.id, issue: 'Missing route'}); continue; }
      let before, previous;
      for (let offset = 1; offset < length; offset += 3) {
        const p = path.getPointAtLength(offset);
        const hit = boxes.find(b => p.x > b.x + .25 && p.x < b.x + b.width - .25 && p.y > b.y + .25 && p.y < b.y + b.height - .25);
        if (hit) { failures.push({edge: path.dataset.id, obstacle: hit.id}); break; }
        if (before && path.dataset.source !== path.dataset.target) {
          const u = {x: previous.x - before.x, y: previous.y - before.y}, v = {x: p.x - previous.x, y: p.y - previous.y};
          const cosine = (u.x * v.x + u.y * v.y) / (Math.hypot(u.x, u.y) * Math.hypot(v.x, v.y));
          if (cosine < .5) { failures.push({edge: path.dataset.id, issue: 'Abrupt bend or hook'}); break; }
        }
        before = previous; previous = p;
      }
    }
    return failures;
  });
  assert.deepEqual(collisions, [], 'Every link must avoid cards and sharp hooks, including hidden similarity links');
}

async function checkMetadataTable(details, values, sources) {
  await details.locator('summary').click();
  await details.locator('table').waitFor();
  assert.deepEqual(await details.locator('th').allTextContents(), ['Name', 'Value', 'Path']);
  const rows = await details.locator('tbody tr').evaluateAll(nodes => nodes.map(node => ({
    key: node.dataset.field, cells: [...node.cells].map(cell => cell.innerText), conflict: node.classList.contains('field-conflict')
  })));
  for (const [key, value] of Object.entries(values)) {
    const entries = Array.isArray(sources[key]) ? sources[key] : [{value, artifact: sources[key] || '—'}];
    const variants = new Map();
    for (const entry of entries) {
      const encoded = JSON.stringify(entry.value);
      if (!variants.has(encoded)) variants.set(encoded, []);
      variants.get(encoded).push(entry.artifact);
    }
    const actual = rows.filter(row => row.key === key);
    assert.equal(actual.length, variants.size);
    for (const [index, [encoded, paths]] of [...variants].entries()) {
      const row = actual[index], stored = JSON.parse(encoded);
      assert.equal(row.cells.length, 3);
      assert.equal(row.cells[1], typeof stored === 'object' ? JSON.stringify(stored) : String(stored));
      assert.equal(row.conflict, variants.size > 1);
      const expectedPaths = [...new Set(paths)];
      if (entries.some(entry => JSON.stringify(entry.value) === encoded && entry.inherited)) expectedPaths.push('Shared parent config (inferred)');
      assert.deepEqual(row.cells[2].split('\n').filter(Boolean), expectedPaths);
    }
  }
  assert(rows.every(row => Object.hasOwn(values, row.key)));
}

(async () => {
  const reportPath = path.resolve(process.argv[2]);
  const report = JSON.parse(readFileSync(path.join(path.dirname(reportPath), 'report.json'), 'utf8'));
  const browser = await chromium.launch({ headless: true, executablePath: process.env.RUN_DIG_BROWSER || undefined });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    const errors = [], remoteRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => { if (/^https?:/.test(request.url())) remoteRequests.push(request.url()); });
    await page.goto(pathToFileURL(reportPath).href);
    const initial = await page.locator('.experiment-link').count();
    assert.equal(initial, report.experiments.length);
    assert(await page.locator('#overview').isVisible());
    assert(await page.locator('#detail').isHidden());
    assert.equal(await page.locator('svg').count(), 0);
    await page.locator('#search').fill('_____no_match_____');
    assert.equal(await page.locator('.experiment-link').count(), 0);
    assert(await page.locator('#empty').isVisible());
    await page.locator('#search').fill('');
    await page.locator('#graph-button').click();
    assert.equal(await page.locator('#global-graph .graph-node').count(), initial);
    assert.equal(await page.locator('#global-graph .graph-edge').count(), report.edges.length);
    assert.equal(await page.locator('#include-similar').isChecked(), true);
    assert(await page.locator('#global-graph .graph-edge.secondary').evaluateAll(nodes => nodes.every(node => Number(getComputedStyle(node).opacity) > 0)), 'Similarity is visible by default alongside source links');
    assert.equal(await page.locator('#global-graph .graph-component, #global-graph .graph-section').count(), 0, 'Experiments share one graph instead of source-only panels and a separate grid');
    assert.equal(await page.locator('#global-graph .graph-edge-label').count(), 0, 'Relation labels do not float across routes; explanations remain in tooltips and details');
    await checkGraphRoutes(page.locator('#global-graph svg'));
    const positions = await page.locator('#global-graph .graph-node').evaluateAll(nodes => nodes.map(node => [node.dataset.id, node.getAttribute('transform')]));
    assert(await page.locator('#global-graph .graph-node text').evaluateAll(nodes => nodes.every(node => node.getComputedTextLength() <= 196.01)), 'Long graph labels stay inside their nodes');
    await page.locator('#global-graph').getByRole('button', {name: 'Fit', exact: true}).click();
    assert(parseInt(await page.locator('#global-graph .graph-tools span').innerText()) >= 75, 'Fit keeps labels readable');
    const rectangles = await page.locator('#global-graph .graph-node rect').evaluateAll(nodes => nodes.map(node => {
      const r = node.getBoundingClientRect(); return {x: r.x, y: r.y, right: r.right, bottom: r.bottom};
    }));
    for (let i = 0; i < rectangles.length; i++) for (const b of rectangles.slice(i + 1)) {
      const a = rectangles[i];
      assert(a.right <= b.x || b.right <= a.x || a.bottom <= b.y || b.bottom <= a.y, 'Graph nodes must not overlap');
    }
    const routes = await page.locator('#global-graph .graph-edge:not(.secondary)').evaluateAll(nodes => nodes.map(node => node.getAttribute('d')));
    assert(routes.every(route => !/NaN|undefined|Infinity/.test(route)));
    assert.equal(new Set(routes).size, routes.length, 'Every relationship has its own route');
    if (routes.length) assert.notEqual(await page.locator('#global-graph .graph-edge').first().evaluate(node => getComputedStyle(node).strokeDasharray), 'none');
    const branching = report.experiments.find(item => report.edges.filter(edge => edge.directed && edge.source === item.id).length > 1);
    if (branching) {
      const starts = await page.locator(`#global-graph .graph-edge:not(.secondary)[data-source="${branching.id}"]`).evaluateAll(nodes => nodes.map(node => node.getAttribute('d').split(' ')[0]));
      assert.equal(new Set(starts).size, starts.length, 'Outgoing links use separate attachment points');
      await page.locator(`#global-graph .graph-node[data-id="${branching.id}"]`).hover();
      assert(await page.locator('#global-graph .graph-node.dimmed').evaluateAll(nodes => nodes.every(node => getComputedStyle(node).opacity === '1' && getComputedStyle(node.querySelector('rect')).fillOpacity === '1')), 'Dimmed cards keep opaque backgrounds');
      assert.equal(await page.locator(`#global-graph .graph-edge.dimmed[data-source="${branching.id}"]`).count(), 0);
      assert.equal(await page.locator(`#global-graph .graph-edge.highlighted:not(.secondary)[data-source="${branching.id}"]`).count(), starts.length);
      assert.equal(await page.locator('#global-graph .graph-edge.highlighted:not(.secondary)').first().evaluate(node => getComputedStyle(node).strokeDasharray), 'none');
      const similar = report.edges.filter(edge => !edge.directed && [edge.source, edge.target].includes(branching.id));
      assert.equal(await page.locator('#global-graph .graph-edge.secondary.highlighted').count(), similar.length, 'Hover reveals every related similarity, not only the top few');
      await page.waitForFunction(() => [...document.querySelectorAll('#global-graph .graph-edge.secondary.highlighted')].every(node => getComputedStyle(node).opacity === '1' && !node.hasAttribute('marker-end')));
      await page.locator('#graph-button').hover();
      assert.equal(await page.locator('#global-graph .dimmed').count(), 0);
      assert.equal(await page.locator('#global-graph .highlighted').count(), 0);
    }
    await page.locator('#include-similar').uncheck();
    await page.waitForFunction(() => [...document.querySelectorAll('#global-graph .graph-edge.secondary')].every(node => getComputedStyle(node).opacity === '0'));
    await page.locator('#include-similar').check();
    assert.equal(await page.locator('#global-graph .graph-edge').count(), report.edges.length);
    await page.waitForFunction(() => [...document.querySelectorAll('#global-graph .graph-edge.secondary')].every(node => Number(getComputedStyle(node).opacity) > 0));
    assert.deepEqual(await page.locator('#global-graph .graph-node').evaluateAll(nodes => nodes.map(node => [node.dataset.id, node.getAttribute('transform')])), positions, 'Showing similarity links must not shuffle the lineage');
    // In an acyclic graph every source must be to the left of its consumer.
    const incoming = new Map(report.experiments.map(item => [item.id, 0])), children = new Map(report.experiments.map(item => [item.id, []]));
    for (const edge of report.edges.filter(edge => edge.directed)) { incoming.set(edge.target, incoming.get(edge.target) + 1); children.get(edge.source).push(edge.target); }
    const queue = report.experiments.filter(item => !incoming.get(item.id)).map(item => item.id);
    for (let i = 0; i < queue.length; i++) for (const child of children.get(queue[i])) { incoming.set(child, incoming.get(child) - 1); if (!incoming.get(child)) queue.push(child); }
    if (queue.length === report.experiments.length) {
      const x = new Map(positions.map(([id, transform]) => [id, Number(transform.match(/translate\(([-.\d]+)/)[1])]));
      for (const edge of report.edges.filter(edge => edge.directed)) assert(x.get(edge.source) < x.get(edge.target), 'Source chains read left to right');
    }
    await page.locator('#global-graph .graph-viewport').evaluate(node => { node.scrollTop = 0; node.scrollLeft = 0; });
    await page.screenshot({ path: process.argv[3] });
    // Exercise cycles, parallel links, long links and similarity-only layouts as well.
    if (report.experiments.length >= 6) for (const scenario of ['cycle', 'fork', 'similarity']) {
      await page.evaluate(scenario => {
        const nodes = experiments.slice(0, 6), pairs = scenario === 'cycle'
          ? [[0, 1], [1, 2], [2, 0], [0, 0], [3, 4]]
          : scenario === 'fork' ? [[0, 1], [1, 2], [2, 3], [0, 3], [0, 3], [4, 3]]
          : [[0, 1], [0, 2], [0, 3], [1, 4], [3, 4]];
        const edges = pairs.map(([a, b], i) => ({id: `check-${i}`, source: nodes[a].id, target: nodes[b].id, directed: scenario !== 'similarity', kind: scenario === 'similarity' ? 'similar_to' : 'references', confidence: .8}));
        const fixture = document.createElement('div'); fixture.id = 'graph-fixture'; fixture.style.width = '840px'; document.body.append(fixture);
        renderGraph(fixture, nodes, edges);
      }, scenario);
      await checkGraphRoutes(page.locator('#graph-fixture svg'));
      if (scenario === 'fork') {
        const separation = await page.locator('#graph-fixture svg').evaluate(svg => {
          const first = svg.querySelector('[data-id="check-3"]'), second = svg.querySelector('[data-id="check-4"]');
          const a = first.getPointAtLength(first.getTotalLength() / 2), b = second.getPointAtLength(second.getTotalLength() / 2);
          return Math.hypot(a.x - b.x, a.y - b.y);
        });
        assert(separation > 2, 'Parallel links remain separated through the middle of their routes');
      }
      await page.locator('#graph-fixture').evaluate(node => node.remove());
    }
    const selected = report.experiments.find(item => item.path === 'leaf') || report.experiments.find(item => report.edges.some(edge => edge.directed && edge.target === item.id)) || report.experiments[0];
    if (selected) {
      await page.locator('#search').fill(selected.path);
      assert(await page.locator(`#global-graph .graph-node[data-id="${selected.id}"]`).evaluate(node => node.classList.contains('graph-match')));
      if (await page.locator('#global-graph .graph-match').count() === 1) {
        assert(await page.locator('#global-graph .graph-match').evaluate(node => {
          const a = node.getBoundingClientRect(), b = node.closest('.graph-viewport').getBoundingClientRect();
          return a.left >= b.left && a.right <= b.right && a.top >= b.top && a.bottom <= b.bottom;
        }), 'A unique search result is scrolled into view');
      }
      await page.locator(`#global-graph .graph-node[data-id="${selected.id}"]`).click();
      assert(await page.locator('#overview').isHidden());
      assert.equal(await page.locator('#experiment-detail h2').innerText(), selected.name);
      const sourceIds = report.edges.filter(edge => edge.directed && edge.target === selected.id).map(edge => edge.id).sort();
      assert.deepEqual((await page.locator('#origin [data-edge-id]').evaluateAll(nodes => nodes.map(node => node.dataset.edgeId))).sort(), sourceIds);
      const expectedChain = new Set();
      for (const direction of ['source', 'target']) {
        const visited = new Set([selected.id]);
        let changed = true;
        while (changed) {
          changed = false;
          for (const edge of report.edges.filter(edge => edge.directed)) {
            if (!visited.has(edge[direction])) continue;
            expectedChain.add(edge.id);
            const next = edge[direction === 'source' ? 'target' : 'source'];
            if (!visited.has(next)) { visited.add(next); changed = true; }
          }
        }
      }
      for (const edge of report.edges) if (!edge.directed && [edge.source, edge.target].includes(selected.id)) expectedChain.add(edge.id);
      assert.deepEqual((await page.locator('#lineage-graph .graph-edge').evaluateAll(nodes => nodes.map(node => node.dataset.id))).sort(), [...expectedChain].sort());
      await checkGraphRoutes(page.locator('#lineage-graph svg'));
      if (sourceIds.length) {
        await page.locator('#origin .relationship summary').first().click();
        await page.locator('#origin .evidence').first().waitFor();
        assert((await page.locator('#origin .evidence').first().innerText()).includes('Confidence:'));
      }
      for (const title of ['Configuration', 'Metrics']) {
        const values = title === 'Configuration' ? selected.config : selected.metrics;
        const sources = title === 'Configuration' ? selected.config_sources : selected.metric_sources;
        if (!Object.keys(values).length) continue;
        const details = page.locator('#experiment-detail details').filter({has: page.locator('summary', {hasText: new RegExp(`^${title} \\(`)})});
        await checkMetadataTable(details, values, sources);
      }
      await page.locator('#back').click();
      assert(await page.locator('#graph-view').isVisible());
      await page.locator('#search').fill('');
      await page.locator('#list-button').click();
      assert.equal(await page.locator('.experiment-link').count(), initial);
    }
    const gitRun = report.experiments.find(item => item.git?.history?.length && item.git?.recorded_commits?.length);
    if (gitRun) {
      await page.locator(`button[data-id="${gitRun.id}"]`).click();
      const details = page.locator('#experiment-detail > details').filter({has: page.getByText('Git history', {exact: true})});
      await details.locator('summary').click();
      assert.deepEqual(await details.locator('th').allTextContents(), ['Commit', 'Message', 'Date']);
      assert.equal(await details.locator('tbody tr').count(), gitRun.git.history.length);
      const text = await details.innerText();
      for (const record of gitRun.git.recorded_commits) assert(text.includes(`${record.artifact}#${record.field}`));
      assert(text.includes('not the code version used to train a checkpoint'));
      await page.setViewportSize({ width: 390, height: 844 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
      await page.setViewportSize({ width: 1440, height: 960 });
      await page.screenshot({path: process.argv[3].replace(/\.png$/, '-git.png')});
      await page.locator('#back').click();
    }
    const unknown = report.experiments.find(item => item.kind !== 'collection' && !report.edges.some(edge => edge.directed && edge.target === item.id));
    if (unknown) {
      await page.locator(`button[data-id="${unknown.id}"]`).click();
      assert((await page.locator('#origin').innerText()).includes('Origin unknown'));
      const similarities = report.edges.filter(edge => !edge.directed && [edge.source, edge.target].includes(unknown.id));
      for (const edge of similarities) assert.equal(await page.locator(`#lineage-graph .graph-edge[data-id="${edge.id}"]`).count(), 1);
      await page.locator('#back').click();
    }
    const conflict = report.experiments.find(item => Object.values(item.config_sources).some(entries => new Set(entries.map(e => JSON.stringify(e.value))).size > 1));
    if (conflict) {
      await page.locator(`button[data-id="${conflict.id}"]`).click();
      assert((await page.locator('.detail-header').innerText()).includes('Conflicting'));
      const details = page.locator('#experiment-detail > details').filter({has: page.locator('summary', {hasText: /^Configuration/})});
      await checkMetadataTable(details, conflict.config, conflict.config_sources);
      await page.locator('#back').click();
    }
    const conflictingSource = report.experiments.find(item => item.references.some(ref => ref.conflict));
    if (conflictingSource) {
      await page.locator(`button[data-id="${conflictingSource.id}"]`).click();
      assert((await page.locator('#origin').innerText()).includes('Source records disagree'));
      assert.equal(await page.locator('#origin [data-edge-id]').count(), report.edges.filter(e => e.directed && e.target === conflictingSource.id).length);
      await page.locator('#back').click();
    }
    for (const warned of report.artifacts.filter(item => item.kind === 'checkpoint' && item.warnings.length).slice(0, 4)) {
      const owner = report.experiments.find(item => item.artifacts.includes(warned.id));
      await page.locator(`button[data-id="${owner.id}"]`).click();
      const notes = await page.locator('.file-note').allTextContents();
      for (const warning of warned.warnings) assert(notes.includes(warning), 'Checkpoint warning appears next to the file');
      await page.locator('#back').click();
    }
    const unresolved = report.experiments.find(item => item.references.some(ref => ref.resolution === 'unresolved'));
    if (unresolved) {
      await page.locator(`button[data-id="${unresolved.id}"]`).click();
      assert((await page.locator('#origin').innerText()).includes('Source not found in this scan'));
      await page.locator('#back').click();
    }
    const checkpoint = report.artifacts.find(item => item.kind === 'checkpoint' && item.metadata?.metric_records?.length);
    if (checkpoint) {
      const owner = report.experiments.find(item => item.artifacts.includes(checkpoint.id));
      await page.locator(`button[data-id="${owner.id}"]`).click();
      const details = page.locator(`[data-checkpoint="${checkpoint.id}"]`);
      await details.locator('summary').click();
      await details.locator('table').first().waitFor();
      assert.equal(await details.locator('table').count(), checkpoint.metadata.metric_records.length);
      for (const [index, record] of checkpoint.metadata.metric_records.entries()) {
        const table = details.locator('table').nth(index);
        assert.deepEqual(await table.locator('th').allTextContents(), ['Name', 'Value', 'Path']);
        assert.equal(await table.locator('tbody tr').count(), Object.keys(record.values).length);
        assert((await table.innerText()).includes(record.source));
        assert((await details.innerText()).includes(record.basis));
      }
      await page.setViewportSize({ width: 390, height: 844 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
      await page.setViewportSize({ width: 1440, height: 960 });
      await page.locator('#back').click();
    }
    const collection = report.experiments.find(item => item.kind === 'collection' && item.children.length);
    if (collection) {
      const row = page.locator('.experiment-row').filter({has: page.locator(`button[data-id="${collection.id}"]`)});
      assert((await row.innerText()).includes(`${collection.subtree_checkpoint_count} total`));
      await row.locator('button').click();
      assert((await page.locator('.detail-header').innerText()).includes(`${collection.subtree_checkpoint_count} checkpoints total`));
      assert.equal(await page.locator('#collection-members [data-child]').count(), collection.children.length);
      assert(!(await page.locator('#experiment-detail').innerText()).includes('No checkpoint files found.'));
      await page.screenshot({path: process.argv[3].replace(/\.png$/, '-collection.png')});
      await page.locator('#collection-members [data-child]').first().click();
      await page.locator(`[data-collection="${collection.id}"]`).click();
      assert.equal(await page.locator('#experiment-detail h2').innerText(), collection.name);
      await page.setViewportSize({ width: 390, height: 844 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
      await page.screenshot({path: process.argv[3].replace(/\.png$/, '-collection-mobile.png')});
      await page.setViewportSize({ width: 1440, height: 960 });
      await page.locator('#back').click();
    }
    const explained = report.edges.find(edge => edge.kind === 'similar_to' && edge.description && Object.keys(edge.changes).length);
    if (explained) {
      await page.locator(`button[data-id="${explained.source}"]`).click();
      await page.locator('#experiment-detail details').filter({has: page.locator('summary', {hasText: /^Similar experiments/})}).locator('summary').click();
      const relation = page.locator(`[data-edge-id="${explained.id}"]`);
      assert((await relation.innerText()).includes(explained.description));
      await relation.locator('summary').click();
      await relation.locator('table').waitFor();
      assert.deepEqual(await relation.locator('th').allTextContents(), ['Name', 'First', 'Second']);
      const rows = await relation.locator('tbody tr').evaluateAll(nodes => nodes.map(node => [...node.cells].map(cell => cell.innerText)));
      assert.deepEqual(rows, Object.entries(explained.changes).map(([key, change]) => [key, String(change.source ?? '(not recorded)'), String(change.target ?? '(not recorded)')]));
      await page.locator('#back').click();
    }
    await page.locator('#notes-label').click();
    await page.locator('#scan-notes p').first().waitFor();
    const download = page.waitForEvent('download');
    await page.getByText('Download JSON', { exact: true }).click();
    assert.equal((await download).suggestedFilename(), 'run-dig-report.json');
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.locator('#search').isVisible());
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    await page.locator('#graph-button').click();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    await page.goto(pathToFileURL(reportPath).href + '#graph');
    assert(await page.locator('#graph-view').isVisible(), 'Demo links open the full graph directly');
    assert(await page.locator('#list-view').isHidden());
    assert.equal(await page.locator('#global-graph .graph-node').count(), initial);
    await page.locator('#global-graph .graph-node').first().click();
    assert(await page.locator('#detail').isVisible());
    await page.locator('#back').click();
    assert(await page.locator('#graph-view').isVisible());
    assert.deepEqual(errors, []);
    assert.deepEqual(remoteRequests, []);
    console.log(JSON.stringify({ experiments: initial, list: 'passed', search: 'passed', sources: 'passed', fullGraph: 'passed', metadataColumns: 'passed', download: 'passed', mobile: 'passed', graphLink: 'passed', errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
