const {chromium} = require('C:/Users/29728/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {pathToFileURL} = require('url');
const path = require('path');
const fs = require('fs');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1080},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(__dirname,'index.html')).href);
 await page.screenshot({path:path.join(__dirname,'preview_desktop.png')});
 const initial=await page.locator('.journal:visible').count();
 await page.selectOption('#tier','重点候选');
 const shortlist=await page.locator('.journal:visible').count();
 await page.click('#reset');await page.fill('#q','Journal of Solar Energy Engineering');
 const search=await page.locator('.journal:visible').count();
 await page.click('#reset');await page.selectOption('#cas','1区');
 const cas1=await page.locator('.journal:visible').count();
 await page.click('#reset');await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:path.join(__dirname,'preview_mobile.png')});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 const result={initial,shortlist,search,cas1,mobile_page_overflow:overflow,page_errors:errors};
 fs.writeFileSync(path.join(__dirname,'layout_check.json'),JSON.stringify(result,null,2));
 console.log(result);await browser.close();
 if(initial!==22||shortlist!==5||search!==1||overflow||errors.length)process.exit(1);
})();

