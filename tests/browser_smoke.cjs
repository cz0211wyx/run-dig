// Optional: node tests/browser_smoke.cjs REPORT_HTML SCREENSHOT.png
const { chromium } = require(process.env.RUN_SHELF_PLAYWRIGHT || 'playwright');
const { pathToFileURL } = require('node:url');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const reportPath = path.resolve(process.argv[2]);
  const report = JSON.parse(readFileSync(path.join(path.dirname(reportPath), 'report.json'), 'utf8'));
  const browser = await chromium.launch({ headless: true, executablePath: process.env.RUN_SHELF_BROWSER || undefined });
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
    assert.equal(await page.locator('#global-graph .graph-edge').count(), report.edges.filter(edge => edge.directed).length);
    await page.locator('#include-similar').check();
    assert.equal(await page.locator('#global-graph .graph-edge').count(), report.edges.length);
    await page.locator('#include-similar').uncheck();
    await page.screenshot({ path: process.argv[3] });
    const selected = report.experiments.find(item => item.path === 'leaf') || report.experiments.find(item => report.edges.some(edge => edge.directed && edge.target === item.id)) || report.experiments[0];
    if (selected) {
      await page.locator('#search').fill(selected.path);
      assert(await page.locator(`#global-graph .graph-node[data-id="${selected.id}"]`).evaluate(node => node.classList.contains('graph-match')));
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
      assert.deepEqual((await page.locator('#lineage-graph .graph-edge').evaluateAll(nodes => nodes.map(node => node.dataset.id))).sort(), [...expectedChain].sort());
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
        await details.locator('summary').click();
        await details.locator('table').waitFor();
        assert.deepEqual(await details.locator('th').allTextContents(), ['Name', 'Value', 'Path']);
        const rows = await details.locator('tbody tr').evaluateAll(nodes => nodes.map(node => [...node.cells].map(cell => cell.innerText)));
        assert.equal(rows.length, Object.keys(values).length);
        for (const [index, key] of Object.keys(values).entries()) {
          assert.equal(rows[index].length, 3);
          const prefix = typeof sources[key] === 'string' ? sources[key].split('/').pop() + ':' : '';
          assert.equal(rows[index][0], prefix && key.startsWith(prefix) ? key.slice(prefix.length) : key);
          assert.equal(rows[index][1], typeof values[key] === 'object' ? JSON.stringify(values[key]) : String(values[key]));
          const paths = Array.isArray(sources[key]) ? sources[key].map(item => item.artifact) : [sources[key]];
          for (const file of paths) assert(rows[index][2].includes(file));
        }
      }
      await page.locator('#back').click();
      assert(await page.locator('#graph-view').isVisible());
      await page.locator('#search').fill('');
      await page.locator('#list-button').click();
      assert.equal(await page.locator('.experiment-link').count(), initial);
    }
    const unknown = report.experiments.find(item => !report.edges.some(edge => edge.directed && edge.target === item.id));
    if (unknown) {
      await page.locator(`button[data-id="${unknown.id}"]`).click();
      assert((await page.locator('#origin').innerText()).includes('Origin unknown'));
      await page.locator('#back').click();
    }
    const unresolved = report.experiments.find(item => item.references.some(ref => ref.resolution === 'unresolved'));
    if (unresolved) {
      await page.locator(`button[data-id="${unresolved.id}"]`).click();
      assert((await page.locator('#origin').innerText()).includes('Source not found in this scan'));
      await page.locator('#back').click();
    }
    await page.locator('#notes-label').click();
    await page.locator('#scan-notes p').first().waitFor();
    const download = page.waitForEvent('download');
    await page.getByText('Download JSON', { exact: true }).click();
    assert.equal((await download).suggestedFilename(), 'run-shelf-report.json');
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.locator('#search').isVisible());
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    await page.locator('#graph-button').click();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    assert.deepEqual(errors, []);
    assert.deepEqual(remoteRequests, []);
    console.log(JSON.stringify({ experiments: initial, list: 'passed', search: 'passed', sources: 'passed', fullGraph: 'passed', metadataColumns: 'passed', download: 'passed', mobile: 'passed', errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
