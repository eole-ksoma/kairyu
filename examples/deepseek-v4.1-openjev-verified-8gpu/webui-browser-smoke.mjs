#!/usr/bin/env node

/**
 * Browser gate for this example's Open WebUI (VCO-D6): both public models are
 * offered; an everyday request on kairyu-verified answers (Jev routes it); a
 * request on kairyu-verified-always answers and its folded internal work
 * shows the "Verification" section with the guarantee outcome.
 */

import { chromium } from 'playwright';

const baseUrl = new URL(process.env.WEBUI_SMOKE_BASE_URL ?? 'http://127.0.0.1:3012');
const actionTimeoutMs = 20_000;
const navigationTimeoutMs = 30_000;
const responseTimeoutMs = Number(process.env.WEBUI_SMOKE_RESPONSE_TIMEOUT_MS ?? 1_800_000);
const models = ['kairyu-verified', 'kairyu-verified-always'];

let browser;
let page;
let currentStep = 'startup';

function invariant(condition, message) {
	if (!condition) throw new Error(message);
}

async function step(name, operation) {
	currentStep = name;
	return operation();
}

async function selectModel(modelId) {
	await page.locator('#model-selector-model-button').click({ timeout: actionTimeoutMs });
	await page.locator('#model-search-input').fill(modelId);
	const option = page.locator(`[role="option"][data-value="${modelId}"]`);
	await option.waitFor({ state: 'visible', timeout: actionTimeoutMs });
	await option.click({ timeout: actionTimeoutMs });
	await page.waitForFunction(
		(id) => document.querySelector('#model-selector-model-button')?.getAttribute('aria-label')?.includes(id),
		modelId,
		{ timeout: actionTimeoutMs }
	);
}

async function send(modelId, prompt) {
	await selectModel(modelId);
	const log = page.locator('ul[role="log"]');
	const before = await log.locator('[role="listitem"]').count();
	await page.locator('#chat-input').fill(prompt);
	await page.locator('#send-message-button').click({ timeout: actionTimeoutMs });
	await page.waitForFunction(
		(count) => document.querySelectorAll('ul[role="log"] [role="listitem"]').length >= count,
		before + 2,
		{ timeout: responseTimeoutMs }
	);
	const item = log.locator('[role="listitem"]').nth(before + 1);
	await item.locator('.copy-response-button').waitFor({ state: 'visible', timeout: responseTimeoutMs });
	const text = (await item.innerText()).trim();
	invariant(text.length > 0, `${modelId}: empty answer`);
	invariant(!/\b(502|error|connection)\b/i.test(text.slice(0, 200)), `${modelId}: visible error: ${text}`);
	return item;
}

async function expandedReasoning(item) {
	const toggle = item.locator('button[aria-expanded]').filter({ hasText: /Thought|Thinking/ });
	invariant((await toggle.count()) >= 1, 'no folded internal-work section');
	await toggle.first().click({ timeout: actionTimeoutMs });
	return (await item.innerText()).trim();
}

async function main() {
	browser = await chromium.launch({ headless: true });
	const context = await browser.newContext({ serviceWorkers: 'block', locale: 'en-US' });
	page = await context.newPage();
	await step('open no-auth chat', async () => {
		await page.goto(baseUrl.href, { waitUntil: 'domcontentloaded', timeout: navigationTimeoutMs });
		await page.locator('#chat-input').waitFor({ state: 'visible', timeout: navigationTimeoutMs });
	});
	await step('both models offered', async () => {
		const ids = await page.evaluate(async () => {
			const response = await fetch('/api/models', {
				headers: { Authorization: `Bearer ${localStorage.token}` }
			});
			return (await response.json()).data.map((model) => model.id).sort();
		});
		invariant(JSON.stringify(ids) === JSON.stringify(models), `models offered: ${JSON.stringify(ids)}`);
	});
	await step('routed model answers an everyday request', async () => {
		await send('kairyu-verified', 'Tell me a fun fact about octopuses.');
	});
	await step('always-verified model shows its verification', async () => {
		const item = await send(
			'kairyu-verified-always',
			'List three primary colors as a comma-separated line, nothing else.'
		);
		const text = await expandedReasoning(item);
		invariant(text.includes('Verification') && /Guaranteed: (yes|no)/.test(text), `no Verification section: ${text.slice(-600)}`);
	});
	console.log('WEBUI BROWSER SMOKE PASS');
}

try {
	await main();
} catch (error) {
	console.error(`WEBUI BROWSER SMOKE FAIL [step=${currentStep}]\n${error?.stack ?? error}`);
	process.exitCode = 1;
} finally {
	await browser?.close().catch(() => {});
}
