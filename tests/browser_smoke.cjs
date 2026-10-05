// Optional viewer smoke test: node tests/browser_smoke.cjs REPORT_HTML SCREENSHOT.png
// Requires Playwright and a browser (or RUN_SHELF_BROWSER pointing to a local Chromium).
const { chromium } = require(process.env.RUN_SHELF_PLAYWRIGHT || 'playwright');
const { pathToFileURL } = require('node:url');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.RUN_SHELF_BROWSER || undefined });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    const errors = [], remoteRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => { if (/^https?:/.test(request.url())) remoteRequests.push(request.url()); });
    await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
    await page.locator('.graph-node').first().waitFor();
    const initial = await page.locator('.node-item').count();
    assert(initial > 0);
    await page.screenshot({ path: process.argv[3], fullPage: true });
    await page.locator('#search').fill('_____no_match_____');
    assert.equal(await page.locator('.node-item').count(), 0);
    await page.locator('#reset').click();
    assert.equal(await page.locator('.node-item').count(), initial);
    await page.locator('.node-item').first().click();
    assert(await page.locator('#inspector').innerText());
    await page.locator('#focus').click();
    assert((await page.locator('.node-item').count()) <= initial);
    await page.locator('#reset').click();
    await page.locator('#audit').click();
    assert(await page.locator('#notes').isVisible());
    await page.locator('#close-notes').click();
    for (const toggle of await page.locator('#legend input').all()) await toggle.uncheck();
    assert.equal(await page.locator('.edge').count(), 0);
    await page.locator('#reset').click();
    if (await page.locator('.edge').count()) {
      const point = await page.locator('.edge').first().evaluate(path => {
        const p = path.getPointAtLength(path.getTotalLength() * .5);
        const screen = new DOMPoint(p.x, p.y).matrixTransform(path.getScreenCTM());
        return { x: screen.x, y: screen.y };
      });
      await page.mouse.click(point.x, point.y);
      assert((await page.locator('#inspector').innerText()).toLowerCase().includes('relationship evidence'));
    }
    const download = page.waitForEvent('download');
    await page.locator('#download').click();
    assert.equal((await download).suggestedFilename(), 'run-shelf-report.json');
    await page.setViewportSize({ width: 600, height: 900 });
    assert(await page.locator('#search').isVisible());
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    assert.deepEqual(errors, []);
    assert.deepEqual(remoteRequests, []);
    console.log(JSON.stringify({ nodes: initial, errors, remoteRequests, search: 'passed', focus: 'passed', edges: 'passed', download: 'passed', narrowViewport: 'passed' }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
