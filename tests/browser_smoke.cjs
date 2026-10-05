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
    await page.screenshot({ path: process.argv[3] });
    await page.locator('#search').fill('_____no_match_____');
    assert.equal(await page.locator('.experiment-link').count(), 0);
    assert(await page.locator('#empty').isVisible());
    await page.locator('#search').fill('');
    const selected = report.experiments.find(item => report.edges.some(edge => edge.source === item.id || edge.target === item.id)) || report.experiments[0];
    if (selected) {
      await page.locator(`button[data-id="${selected.id}"]`).click();
      assert(await page.locator('#overview').isHidden());
      assert.equal(await page.locator('#experiment-detail h2').innerText(), selected.name);
      assert.equal(await page.locator('svg').count(), 0);
      if (await page.locator('.relationship').count()) {
        await page.locator('.relationship summary').first().click();
        await page.locator('.relationship .evidence').first().waitFor();
        assert((await page.locator('.relationship .evidence').first().innerText()).includes('Confidence:'));
        await page.getByText('Show relationship graph', { exact: true }).click();
        await page.locator('svg .graph-node').first().waitFor();
        assert((await page.locator('svg .graph-node').count()) > 1);
        await page.locator('.related-link').first().click();
        assert.notEqual(await page.locator('#experiment-detail h2').innerText(), '');
      }
      await page.locator('#back').click();
      assert.equal(await page.locator('.experiment-link').count(), initial);
    }
    await page.locator('#notes-label').click();
    await page.locator('#scan-notes p').first().waitFor();
    const download = page.waitForEvent('download');
    await page.getByText('Download JSON', { exact: true }).click();
    assert.equal((await download).suggestedFilename(), 'run-shelf-report.json');
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.locator('#search').isVisible());
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    assert.deepEqual(errors, []);
    assert.deepEqual(remoteRequests, []);
    console.log(JSON.stringify({ experiments: initial, list: 'passed', search: 'passed', details: 'passed', optionalGraph: 'passed', download: 'passed', mobile: 'passed', errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
