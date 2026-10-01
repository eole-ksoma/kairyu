#!/usr/bin/env node

/**
 * Browser gate for this example's Open WebUI v0.11.0 surface (DTO-D16/D17).
 *
 * Checks the one public product model, then sends one chat and requires the
 * folded, attributed internal-work item and a separate final answer. Both
 * routes publish from DeepSeek-V4.1; the ensemble also shows its Qwen stages.
 */

import { chromium } from 'playwright';

const baseUrl = new URL(process.env.WEBUI_SMOKE_BASE_URL ?? 'http://127.0.0.1:3009');
const browserWsEndpoint = process.env.WEBUI_SMOKE_BROWSER_WS_ENDPOINT ?? '';
const chromiumExecutable = process.env.WEBUI_SMOKE_CHROMIUM_EXECUTABLE ?? '';
const productModel = 'kairyu-auto-max';
const models = Object.freeze([productModel]);

const actionTimeoutMs = positiveIntegerEnv('WEBUI_SMOKE_ACTION_TIMEOUT_MS', 20_000);
const navigationTimeoutMs = positiveIntegerEnv('WEBUI_SMOKE_NAVIGATION_TIMEOUT_MS', 30_000);
const responseTimeoutMs = positiveIntegerEnv('WEBUI_SMOKE_RESPONSE_TIMEOUT_MS', 1_800_000);
const phaseTimeoutMs = positiveIntegerEnv('WEBUI_SMOKE_PHASE_TIMEOUT_MS', 2_100_000);

let browser;
let context;
let page;
let currentStep = 'startup';
const sameOriginRequestFailures = [];

function positiveIntegerEnv(name, fallback) {
	const raw = process.env[name];
	if (raw === undefined || raw === '') {
		return fallback;
	}
	const parsed = Number(raw);
	if (!Number.isSafeInteger(parsed) || parsed <= 0) {
		throw new Error(`${name} must be a positive integer number of milliseconds, got ${raw}`);
	}
	return parsed;
}

function invariant(condition, message) {
	if (!condition) {
		throw new Error(message);
	}
}

async function step(name, operation) {
	currentStep = name;
	try {
		return await operation();
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		// Overlay toasts/popups are the most common cause of intercepted
		// clicks; include their visible text so a failure names the actual
		// blocker instead of a bare locator timeout.
		let overlayNote = '';
		try {
			const overlays = await page.evaluate(() =>
				Array.from(document.querySelectorAll('div.z-50'))
					.map((node) => node.innerText.trim().replace(/\s+/g, ' ').slice(0, 200))
					.filter(Boolean)
					.slice(0, 3)
			);
			if (overlays.length > 0) {
				overlayNote = `\n[z-50 overlays: ${JSON.stringify(overlays)}]`;
			}
		} catch {
			// diagnostics never mask the original failure
		}
		throw new Error(`${name}: ${detail}${overlayNote}`, { cause: error });
	}
}

async function bounded(promise, timeoutMs, description) {
	let timer;
	try {
		return await Promise.race([
			promise,
			new Promise((_, reject) => {
				timer = setTimeout(
					() => reject(new Error(`${description} exceeded ${timeoutMs} ms`)),
					timeoutMs
				);
			})
		]);
	} finally {
		clearTimeout(timer);
	}
}

async function becomesVisible(locator, timeoutMs) {
	return locator
		.waitFor({ state: 'visible', timeout: timeoutMs })
		.then(() => true)
		.catch(() => false);
}

function isAllowedBrowserUrl(rawUrl) {
	const url = new URL(rawUrl);
	if (['about:', 'blob:', 'data:'].includes(url.protocol)) {
		return true;
	}
	if (!['http:', 'https:', 'ws:', 'wss:'].includes(url.protocol)) {
		return false;
	}
	return url.host === baseUrl.host;
}

async function launchBrowser() {
	if (browserWsEndpoint) {
		return chromium.connect(browserWsEndpoint, { timeout: navigationTimeoutMs });
	}
	return chromium.launch({
		headless: true,
		...(chromiumExecutable ? { executablePath: chromiumExecutable } : {})
	});
}

async function installNetworkGuard(browserContext) {
	await browserContext.route('**/*', async (route) => {
		const requestUrl = route.request().url();
		if (isAllowedBrowserUrl(requestUrl)) {
			await route.continue();
		} else {
			await route.abort('blockedbyclient');
		}
	});
}

async function goto(pathname) {
	const target = new URL(pathname, baseUrl);
	await page.goto(target.href, {
		waitUntil: 'domcontentloaded',
		timeout: navigationTimeoutMs
	});
}

async function waitForChatSurface() {
	const selector = page.locator('#model-selector-model-button');
	await selector.waitFor({ state: 'visible', timeout: navigationTimeoutMs });
	await page.locator('#chat-input').waitFor({ state: 'visible', timeout: navigationTimeoutMs });
}

async function selectModel(modelId) {
	invariant(models.includes(modelId), `unsupported smoke model ${modelId}`);
	const trigger = page.locator('#model-selector-model-button');
	await trigger.waitFor({ state: 'visible', timeout: actionTimeoutMs });
	await trigger.click({ timeout: actionTimeoutMs });

	const search = page.locator('#model-search-input');
	await search.waitFor({ state: 'visible', timeout: actionTimeoutMs });
	await search.fill(modelId);

	const option = page.locator(`[role="option"][data-value="${modelId}"]`);
	await option.waitFor({ state: 'visible', timeout: actionTimeoutMs });
	const optionLabel = await option.getAttribute('aria-label');
	invariant(
		optionLabel?.startsWith('Select ') && optionLabel.endsWith(' model'),
		`model option ${modelId} did not expose the pinned accessible label; got ${optionLabel}`
	);
	await option.click({ timeout: actionTimeoutMs });

	await page.waitForFunction(
		(id) =>
			document
				.querySelector('#model-selector-model-button')
				?.getAttribute('aria-label')
				?.includes(id),
		modelId,
		{ timeout: actionTimeoutMs }
	);
}

async function browserJson(pathname, options = {}) {
	const result = await page.evaluate(
		async ({ pathname: path, options: requestOptions }) => {
			const token = localStorage.token;
			if (!token) {
				return { error: 'browser session has no localStorage token' };
			}
			const headers = new Headers(requestOptions.headers ?? {});
			headers.set('Authorization', `Bearer ${token}`);
			if (
				requestOptions.body !== undefined &&
				typeof requestOptions.body === 'string' &&
				!headers.has('Content-Type')
			) {
				headers.set('Content-Type', 'application/json');
			}
			const response = await fetch(path, {
				...requestOptions,
				headers
			});
			const text = await response.text();
			let body;
			try {
				body = JSON.parse(text);
			} catch {
				body = text;
			}
			return {
				status: response.status,
				contentType: response.headers.get('content-type') ?? '',
				body,
				text
			};
		},
		{ pathname, options }
	);
	invariant(!result.error, result.error);
	return result;
}

async function sendUiMessage(modelId, marker, prompt = `Reply with this exact local marker: ${marker}`) {
	await selectModel(modelId);
	const log = page.locator('ul[role="log"]');
	const before = await log.locator('[role="listitem"]').count();

	const input = page.locator('#chat-input');
	await input.fill(prompt);
	const responsePromise = page.waitForResponse(
		(response) => {
			const request = response.request();
			if (
				new URL(response.url()).pathname !== '/api/chat/completions' ||
				request.method() !== 'POST'
			) {
				return false;
			}
			try {
				const body = request.postDataJSON();
				return JSON.stringify(body).includes(marker);
			} catch {
				return false;
			}
		},
		{ timeout: responseTimeoutMs }
	);
	await page.locator('#send-message-button').click({ timeout: actionTimeoutMs });
	const networkResponse = await bounded(
		responsePromise,
		responseTimeoutMs,
		`${modelId} UI chat network response`
	);
	const networkBody = await bounded(
		networkResponse.text(),
		responseTimeoutMs,
		`${modelId} UI chat network response body`
	);
	const requestPath = new URL(networkResponse.url()).pathname;
	const requestBody = networkResponse.request().postDataJSON();
	invariant(
		requestPath === '/api/chat/completions',
		`${modelId} UI chat used unexpected path ${requestPath}`
	);
	invariant(
		requestBody?.model === modelId && requestBody?.stream === true,
		`${modelId} UI chat request had unexpected model/stream: ${JSON.stringify({
			model: requestBody?.model,
			stream: requestBody?.stream
		})}`
	);
	invariant(
		networkResponse.status() === 200,
		`${modelId} UI chat returned HTTP ${networkResponse.status()}`
	);
	const contentType = networkResponse.headers()['content-type'] ?? '';
	invariant(
		contentType.toLowerCase().includes('application/json'),
		`${modelId} UI chat response content-type was ${contentType}`
	);
	let taskReceipt;
	try {
		taskReceipt = JSON.parse(networkBody);
	} catch (error) {
		throw new Error(`${modelId} UI chat returned invalid task JSON: ${error.message}`);
	}
	invariant(
		taskReceipt?.status === true &&
			Array.isArray(taskReceipt.task_ids) &&
			taskReceipt.task_ids.length === 1 &&
			typeof taskReceipt.task_ids[0] === 'string' &&
			taskReceipt.task_ids[0].length > 0 &&
			typeof taskReceipt.chat_id === 'string' &&
			taskReceipt.chat_id.length > 0,
		`${modelId} UI chat returned an invalid task receipt: ${networkBody}`
	);

	await page.waitForFunction(
		(expectedCount) =>
			document.querySelectorAll('ul[role="log"] [role="listitem"]').length >= expectedCount,
		before + 2,
		{ timeout: responseTimeoutMs }
	);

	const responseItem = log.locator('[role="listitem"]').nth(before + 1);
	await responseItem.locator('.copy-response-button').waitFor({
		state: 'visible',
		timeout: responseTimeoutMs
	});
	const responseText = (await responseItem.innerText()).trim();
	invariant(
		responseText.length > 0,
		`${modelId} UI chat completed without visible assistant content`
	);
	invariant(
		!/error|issue|server|connection|provider|upstream|not found|502/i.test(responseText),
		`${modelId} UI chat completed with a visible error: ${responseText}`
	);
	return responseItem;
}

async function assertTieredProductInventory() {
	const inventory = await browserJson('/api/models');
	const modelConfig = await browserJson('/api/v1/configs/models');
	invariant(
		inventory.status === 200 && Array.isArray(inventory.body?.data),
		`Open WebUI model inventory failed: HTTP ${inventory.status}: ${inventory.text}`
	);
	const modelIds = inventory.body.data.map((model) => model?.id).filter(Boolean).sort();
	invariant(
		JSON.stringify(modelIds) === JSON.stringify([productModel]),
		`Open WebUI must expose only ${productModel}; got ${JSON.stringify(modelIds)}`
	);
	invariant(
		modelConfig.status === 200 && modelConfig.body?.DEFAULT_MODEL_PARAMS?.stream_response === false,
		`Open WebUI effective model params did not disable streaming: ${modelConfig.text}`
	);
	await selectModel(productModel);
}

async function assertTieredReasoningUi() {
	const prompt = 'PAC1 antagonistのMOAを簡潔に説明してください';
	const responseItem = await sendUiMessage(
		productModel,
		prompt,
		prompt
	);
	const reasoningToggle = responseItem.locator('button[aria-expanded]').filter({
		hasText: /Thought|Thinking/
	});
	invariant(
		(await reasoningToggle.count()) === 1,
		`expected one reasoning toggle in the assistant answer, got ${await reasoningToggle.count()}`
	);
	invariant(
		(await reasoningToggle.getAttribute('aria-expanded')) === 'false',
		'intermediate processing was not initially folded'
	);

	await reasoningToggle.click({ timeout: actionTimeoutMs });
	await page.waitForFunction(
		(button) => button.getAttribute('aria-expanded') === 'true',
		await reasoningToggle.elementHandle(),
		{ timeout: actionTimeoutMs }
	);

	const separation = await responseItem.evaluate((item) => {
		const toggle = [...item.querySelectorAll('button[aria-expanded]')].find((button) =>
			/Thought|Thinking/.test(button.textContent ?? '')
		);
		if (!toggle || !toggle.parentElement) {
			return { error: 'reasoning toggle/root not found' };
		}
		const reasoningRoot = toggle.parentElement;
		const outputRoot = reasoningRoot.parentElement;
		if (!outputRoot) {
			return { error: 'assistant output root not found' };
		}
		const finalRoots = [...outputRoot.children].filter(
			(child) => child !== reasoningRoot && child.classList.contains('markdown-prose')
		);
		return {
			reasoning: reasoningRoot.innerText.trim(),
			finals: finalRoots.map((root) => root.innerText.trim()).filter(Boolean),
			distinctNodes: finalRoots.every((root) => root !== reasoningRoot)
		};
	});
	invariant(!separation.error, separation.error);
	invariant(
		separation.distinctNodes && separation.finals.length === 1,
		`reasoning and final answer were not separate sibling sections: ${JSON.stringify(separation)}`
	);

	for (const label of ['L2 role:', 'L1 worker:', 'Engine:', 'Model:']) {
		invariant(
			separation.reasoning.includes(label),
			`expanded reasoning did not show ${label}: ${separation.reasoning}`
		);
	}
	// Both routes publish from DeepSeek-V4.1 (tier2); the ensemble also shows
	// its Qwen (tier1) stages, the deepseek_think route does not (DTO-D17).
	const identities = ['Final answer attribution', 'publisher', 'tier2', 'deepseek-v4.1-flash'];
	if (separation.reasoning.includes('tier1')) {
		identities.push('qwen3.8-27b');
	}
	for (const identity of identities) {
		invariant(
			separation.reasoning.includes(identity),
			`expanded reasoning did not attribute ${identity}: ${separation.reasoning}`
		);
	}
	const finalText = separation.finals[0];
	invariant(finalText.length > 0, 'L3 final-answer section was empty');
	invariant(
		!['L2 role:', 'L1 worker:', 'Engine:', 'Model:'].some((label) => finalText.includes(label)),
		`L3 final answer mixed in orchestration attribution: ${finalText}`
	);
}

async function main() {
	invariant(
		['http:', 'https:'].includes(baseUrl.protocol),
		`WEBUI_SMOKE_BASE_URL must be HTTP(S), got ${baseUrl.href}`
	);
	invariant(baseUrl.pathname === '/', `WEBUI_SMOKE_BASE_URL must not contain a path: ${baseUrl.href}`);

	browser = await step('launch Playwright Chromium', launchBrowser);
	context = await step('create isolated browser context', () =>
		browser.newContext({
			serviceWorkers: 'block',
			locale: 'en-US',
			timezoneId: 'UTC'
		})
	);
	await step('install same-origin browser network guard', () => installNetworkGuard(context));
	page = await context.newPage();
	page.setDefaultTimeout(actionTimeoutMs);
	page.setDefaultNavigationTimeout(navigationTimeoutMs);
	page.on('requestfailed', (request) => {
		if (isAllowedBrowserUrl(request.url())) {
			sameOriginRequestFailures.push(
				`${request.method()} ${request.url()} (${request.failure()?.errorText ?? 'unknown error'})`
			);
		}
	});

	await step('open no-auth chat', async () => {
		await goto('/');
		await waitForChatSurface();
	});
	await step('single product model inventory', assertTieredProductInventory);
	await step('folded attributed reasoning and separate final answer', assertTieredReasoningUi);

	console.log('WEBUI BROWSER SMOKE PASS');
}

try {
	await bounded(main(), phaseTimeoutMs, 'browser smoke');
} catch (error) {
	const detail = error instanceof Error ? error.stack ?? error.message : String(error);
	console.error(`WEBUI BROWSER SMOKE FAIL [step=${currentStep}]\n${detail}`);
	if (page) {
		console.error(`Current URL: ${page.url()}`);
		const visibleText = await page
			.locator('body')
			.innerText({ timeout: 2_000 })
			.catch(() => '');
		if (visibleText) {
			console.error(`Visible page text (tail):\n${visibleText.slice(-2_000)}`);
		}
	}
	if (sameOriginRequestFailures.length > 0) {
		console.error(
			`Recent same-origin request failures:\n${sameOriginRequestFailures.slice(-10).join('\n')}`
		);
	}
	process.exitCode = 1;
} finally {
	await context?.close().catch(() => {});
	await browser?.close().catch(() => {});
}
