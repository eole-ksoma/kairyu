#!/usr/bin/env node

/**
 * Browser gate for the pinned Open WebUI v0.11.0 compose surface.
 *
 * The surrounding shell smoke owns service lifecycle and invokes this script
 * three times:
 *
 *   WEBUI_SMOKE_PHASE=initial  node scripts/webui_browser_smoke.mjs
 *   WEBUI_SMOKE_PHASE=outage   node scripts/webui_browser_smoke.mjs
 *   WEBUI_SMOKE_PHASE=recovery node scripts/webui_browser_smoke.mjs
 *   WEBUI_SMOKE_PHASE=tiered   node scripts/webui_browser_smoke.mjs
 *
 * `initial` requires a fresh Open WebUI data volume. `outage` runs while
 * Kairyu is stopped, and `recovery` runs after Kairyu is started again without
 * restarting Open WebUI.
 */

import { chromium } from 'playwright';

const PHASES = new Set(['initial', 'outage', 'recovery', 'tiered']);
const phase = process.env.WEBUI_SMOKE_PHASE ?? 'initial';
const baseUrl = new URL(
	process.env.WEBUI_SMOKE_BASE_URL ??
		process.env.WEBUI_BASE_URL ??
		'http://127.0.0.1:3000'
);
const email = process.env.WEBUI_SMOKE_EMAIL ?? 'kairyu-smoke@example.invalid';
const password = process.env.WEBUI_SMOKE_PASSWORD ?? 'Kairyu-WebUI-Smoke-197!';
const fullName = process.env.WEBUI_SMOKE_NAME ?? 'Kairyu Browser Smoke';
const browserWsEndpoint = process.env.WEBUI_SMOKE_BROWSER_WS_ENDPOINT ?? '';
const chromiumExecutable = process.env.WEBUI_SMOKE_CHROMIUM_EXECUTABLE ?? '';
const directModel = process.env.EXPECTED_DIRECT_MODEL ?? 'default';
const autoModel = process.env.EXPECTED_AUTO_MODEL ?? 'kairyu-auto';
const productModel = process.env.EXPECTED_PRODUCT_MODEL ?? 'kairyu-auto-max';

const actionTimeoutMs = positiveIntegerEnv('WEBUI_SMOKE_ACTION_TIMEOUT_MS', 20_000);
const navigationTimeoutMs = positiveIntegerEnv('WEBUI_SMOKE_NAVIGATION_TIMEOUT_MS', 30_000);
const responseTimeoutMs = positiveIntegerEnv('WEBUI_SMOKE_RESPONSE_TIMEOUT_MS', 45_000);
const phaseTimeoutMs = positiveIntegerEnv('WEBUI_SMOKE_PHASE_TIMEOUT_MS', 120_000);

const models = Object.freeze(phase === 'tiered' ? [productModel] : [directModel, autoModel]);
const initialDefaultMarker = 'KAIRYU_WEBUI_INITIAL_DEFAULT_197';
// AUTO wraps the query in a prompt-injection boundary before the deterministic
// mock echoes its final 48 characters. Keep the SSE marker plus `_SSE` within
// the 16 characters that remain ahead of that boundary.
const initialAutoMarker = 'KA_AUTO_197';
const reconnectMarker = 'KAIRYU_WEBUI_RECONNECT_197';
const outageMarker = 'KAIRYU_WEBUI_OUTAGE_197';
const recoveryMarker = 'KAIRYU_WEBUI_RECOVERY_197';
const ragFilename = 'kairyu-rag-202.txt';
const ragNeedle = 'KAIRYU_RAG_NEEDLE_202';
const ragAnswer = 'KAIRYU_RAG_ANSWER_202';
const ragQuery = 'What is the archive code in the local deployment record?';

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

async function dismissFreshInstallChangelog() {
	const accept = page.getByRole('button', {
		name: "Okay, Let's Go!",
		exact: true
	});
	if (await becomesVisible(accept, 5_000)) {
		await accept.click({ timeout: actionTimeoutMs });
	}
}

async function signupFreshUser() {
	await goto('/auth');

	const getStarted = page.getByRole('button', { name: 'Get started', exact: true });
	if (await becomesVisible(getStarted, 5_000)) {
		await getStarted.click({ timeout: actionTimeoutMs });
	}

	const signupCard = page.locator('#auth-login-card');
	await signupCard.waitFor({ state: 'visible', timeout: navigationTimeoutMs });
	const nameInput = page.locator('#name');
	invariant(
		await becomesVisible(nameInput, 5_000),
		'fresh signup form was not shown; ensure the Open WebUI data volume is empty and auth is enabled'
	);

	await nameInput.fill(fullName);
	await page.locator('#email').fill(email);
	await page.locator('#password').fill(password);

	const confirmation = page.locator('#confirm-password');
	if (await becomesVisible(confirmation, 1_000)) {
		await confirmation.fill(password);
	}

	const submit = page.getByRole('button', {
		name: /^(Create Admin Account|Create Account)$/,
		exact: true
	});
	await submit.click({ timeout: actionTimeoutMs });
	await dismissFreshInstallChangelog();
	await waitForChatSurface();

	const hasToken = await page.evaluate(() => Boolean(localStorage.token));
	invariant(hasToken, 'signup reached the chat surface without an authenticated browser session');
}

async function signinExistingUser() {
	await goto('/auth?form=1');
	await page.locator('#auth-login-card').waitFor({ state: 'visible', timeout: navigationTimeoutMs });

	await page.locator('#email').fill(email);
	await page.locator('#password').fill(password);
	await page.getByRole('button', { name: 'Sign in', exact: true }).click({
		timeout: actionTimeoutMs
	});
	await page.locator('#main-content').waitFor({ state: 'attached', timeout: navigationTimeoutMs });

	const hasToken = await page.evaluate(() => Boolean(localStorage.token));
	invariant(hasToken, 'sign-in completed without an authenticated browser session');
}

async function findPersistedSmokeChat() {
	const result = await page.evaluate(async (searchText) => {
		const token = localStorage.token;
		if (!token) {
			return { ok: false, error: 'browser session has no localStorage token' };
		}

		const response = await fetch(
			`/api/v1/chats/search?text=${encodeURIComponent(searchText)}&page=1`,
			{
				headers: {
					Accept: 'application/json',
					Authorization: `Bearer ${token}`
				}
			}
		);
		const body = await response.text();
		if (!response.ok) {
			return { ok: false, error: `chat search returned HTTP ${response.status}: ${body}` };
		}
		let chats;
		try {
			chats = JSON.parse(body);
		} catch {
			return { ok: false, error: `chat search returned invalid JSON: ${body}` };
		}
		if (!Array.isArray(chats) || chats.length === 0 || typeof chats[0]?.id !== 'string') {
			return { ok: false, error: `no persisted smoke chat contains ${searchText}` };
		}
		return { ok: true, id: chats[0].id };
	}, initialDefaultMarker);

	invariant(result.ok, result.error);
	return result.id;
}

async function openPersistedSmokeChat() {
	const chatId = await findPersistedSmokeChat();
	await goto(`/c/${encodeURIComponent(chatId)}`);
	await waitForChatSurface();
	await page.getByText(initialDefaultMarker, { exact: false }).first().waitFor({
		state: 'visible',
		timeout: navigationTimeoutMs
	});
	return chatId;
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

async function assertModelsSelectable() {
	for (const modelId of models) {
		await selectModel(modelId);
	}
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

async function uploadRagDocument() {
	const document = [
		'Kairyu local deployment record.',
		'The archive code is stored only in this uploaded document.',
		`Archive code: ${ragNeedle}`
	].join('\n');
	invariant(!ragQuery.includes(ragNeedle), 'RAG query leaked the retrieval needle');
	invariant(!ragQuery.includes(ragAnswer), 'RAG query leaked the canned answer');

	const upload = await page.evaluate(
		async ({ filename, content }) => {
			const token = localStorage.token;
			if (!token) {
				return { error: 'browser session has no localStorage token' };
			}
			const form = new FormData();
			form.append(
				'file',
				new File([content], filename, { type: 'text/plain' })
			);
			const response = await fetch(
				'/api/v1/files/?process=true&process_in_background=false',
				{
					method: 'POST',
					headers: { Authorization: `Bearer ${token}` },
					body: form
				}
			);
			const text = await response.text();
			let body;
			try {
				body = JSON.parse(text);
			} catch {
				body = text;
			}
			return { status: response.status, body, text };
		},
		{ filename: ragFilename, content: document }
	);
	invariant(!upload.error, upload.error);
	invariant(
		upload.status === 200 && typeof upload.body?.id === 'string',
		`RAG document upload failed: HTTP ${upload.status}: ${upload.text}`
	);

	const fileId = upload.body.id;
	for (let attempt = 0; attempt < 40; attempt += 1) {
		const status = await browserJson(
			`/api/v1/files/${encodeURIComponent(fileId)}/process/status`
		);
		invariant(
			status.status === 200,
			`RAG document status failed: HTTP ${status.status}: ${status.text}`
		);
		if (status.body?.status === 'completed') {
			return fileId;
		}
		if (status.body?.status === 'failed') {
			throw new Error(`RAG document processing failed: ${status.text}`);
		}
		await new Promise((resolve) => setTimeout(resolve, 250));
	}
	throw new Error('RAG document processing did not complete within 10 seconds');
}

async function queryRagDocument(fileId) {
	const retrieval = await browserJson('/api/v1/retrieval/query/doc', {
		method: 'POST',
		body: JSON.stringify({
			collection_name: `file-${fileId}`,
			query: ragQuery,
			k: 1,
			hybrid: false
		})
	});
	invariant(
		retrieval.status === 200,
		`RAG retrieval failed: HTTP ${retrieval.status}: ${retrieval.text}`
	);
	invariant(
		retrieval.text.includes(ragNeedle),
		`RAG retrieval did not return the uploaded needle: ${retrieval.text}`
	);
	return retrieval.body;
}

async function answerWithRagDocument(fileId) {
	const completion = await browserJson('/api/chat/completions', {
		method: 'POST',
		body: JSON.stringify({
			model: directModel,
			stream: false,
			messages: [{ role: 'user', content: ragQuery }],
			files: [{ type: 'file', id: fileId, name: ragFilename }]
		})
	});
	invariant(
		completion.status === 200,
		`RAG chat failed: HTTP ${completion.status}: ${completion.text}`
	);
	const answer = completion.body?.choices?.[0]?.message?.content;
	invariant(
		typeof answer === 'string' && answer.includes(ragAnswer),
		`RAG chat did not produce the retrieval-only answer: ${completion.text}`
	);
	invariant(
		answer.includes('[1]'),
		`RAG chat answer did not preserve its retrieved-source citation: ${answer}`
	);
}

async function assertRagEndToEnd() {
	const fileId = await uploadRagDocument();
	await queryRagDocument(fileId);
	await answerWithRagDocument(fileId);
}

async function findPersistedRagDocument() {
	const search = await browserJson(
		`/api/v1/files/search?filename=${encodeURIComponent(ragFilename)}`
	);
	invariant(
		search.status === 200 && Array.isArray(search.body),
		`persisted RAG document search failed: HTTP ${search.status}: ${search.text}`
	);
	const file = search.body.find(
		(item) => item?.filename === ragFilename && typeof item?.id === 'string'
	);
	invariant(file, `persisted RAG document ${ragFilename} was not found`);
	return file.id;
}

async function assertRagAfterGatewayRestart() {
	const fileId = await findPersistedRagDocument();
	await queryRagDocument(fileId);
	await answerWithRagDocument(fileId);
}

function parseSse(body, modelId) {
	const payloads = [];
	let sawDone = false;
	for (const rawLine of body.split(/\r?\n/)) {
		if (!rawLine.startsWith('data:')) {
			continue;
		}
		const data = rawLine.slice('data:'.length).trim();
		if (data === '[DONE]') {
			sawDone = true;
			continue;
		}
		if (!data) {
			continue;
		}
		try {
			payloads.push(JSON.parse(data));
		} catch (error) {
			throw new Error(
				`${modelId} SSE contained invalid JSON data ${JSON.stringify(data)}: ${error.message}`
			);
		}
	}

	const contentDeltas = payloads
		.flatMap((payload) => payload?.choices ?? [])
		.map((choice) => choice?.delta?.content)
		.filter((content) => typeof content === 'string' && content.length > 0);

	const streamError = payloads.find((payload) => payload?.error !== undefined);
	invariant(
		streamError === undefined,
		`${modelId} browser-visible SSE contained an error event: ${JSON.stringify(streamError?.error)}`
	);
	invariant(sawDone, `${modelId} browser-visible SSE body did not contain data: [DONE]`);
	invariant(
		contentDeltas.length >= 2,
		`${modelId} browser-visible SSE contained ${contentDeltas.length} content delta(s), expected at least 2`
	);
	return contentDeltas;
}

async function browserProxyRequest(modelId, marker) {
	const responsePromise = page.waitForResponse(
		(response) => {
			if (new URL(response.url()).pathname !== '/openai/chat/completions') {
				return false;
			}
			if (response.request().method() !== 'POST') {
				return false;
			}
			try {
				const body = response.request().postDataJSON();
				return body?.model === modelId && body?.stream === true;
			} catch {
				return false;
			}
		},
		{ timeout: responseTimeoutMs }
	);

	const browserResultPromise = page.evaluate(
		async ({ model, prompt }) => {
			const token = localStorage.token;
			if (!token) {
				return { error: 'browser session has no localStorage token' };
			}
			const response = await fetch('/openai/chat/completions', {
				method: 'POST',
				headers: {
					Accept: 'text/event-stream',
					Authorization: `Bearer ${token}`,
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					model,
					stream: true,
					messages: [
						{
							role: 'user',
							content: `Stream this exact local marker in the response: ${prompt}`
						}
					]
				})
			});

			const reader = response.body?.getReader();
			const decoder = new TextDecoder();
			let body = '';
			let chunks = 0;
			if (reader) {
				while (true) {
					const { done, value } = await reader.read();
					if (done) {
						break;
					}
					chunks += 1;
					body += decoder.decode(value, { stream: true });
				}
				body += decoder.decode();
			} else {
				body = await response.text();
			}
			return {
				status: response.status,
				contentType: response.headers.get('content-type') ?? '',
				body,
				chunks
			};
		},
		{ model: modelId, prompt: marker }
	);

	const [networkResponse, browserResult] = await bounded(
		Promise.all([responsePromise, browserResultPromise]),
		responseTimeoutMs,
		`${modelId} browser proxy request`
	);
	const networkBody = await bounded(
		networkResponse.text(),
		responseTimeoutMs,
		`${modelId} Playwright network response body`
	);
	return { networkResponse, networkBody, browserResult };
}

async function assertBrowserStreaming(modelId, marker) {
	const { networkResponse, networkBody, browserResult } = await browserProxyRequest(
		modelId,
		marker
	);
	invariant(!browserResult.error, browserResult.error);
	invariant(
		networkResponse.status() === 200 && browserResult.status === 200,
		`${modelId} stream returned HTTP ${networkResponse.status()}/${browserResult.status}`
	);

	const playwrightContentType = networkResponse.headers()['content-type'] ?? '';
	invariant(
		playwrightContentType.toLowerCase().includes('text/event-stream'),
		`${modelId} Playwright response content-type was ${playwrightContentType}`
	);
	invariant(
		browserResult.contentType.toLowerCase().includes('text/event-stream'),
		`${modelId} browser fetch content-type was ${browserResult.contentType}`
	);
	invariant(
		browserResult.chunks >= 1,
		`${modelId} browser ReadableStream produced no network chunks`
	);

	const browserDeltas = parseSse(browserResult.body, modelId);
	const playwrightDeltas = parseSse(networkBody, modelId);
	invariant(
		browserDeltas.join('').includes(marker),
		`${modelId} browser SSE body did not contain marker ${marker}`
	);
	invariant(
		playwrightDeltas.join('').includes(marker),
		`${modelId} Playwright response body did not contain marker ${marker}`
	);
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

async function assertReloadReconnectsAndPreservesChat() {
	const chatUrl = new URL(page.url());
	invariant(
		/^\/c\/[^/]+$/.test(chatUrl.pathname),
		`completed UI chat was not persisted at a /c/<id> URL; got ${chatUrl.pathname}`
	);

	const websocketPromise = page.waitForEvent('websocket', {
		predicate: (socket) => new URL(socket.url()).pathname === '/ws/socket.io/',
		timeout: navigationTimeoutMs
	});
	await page.reload({ waitUntil: 'domcontentloaded', timeout: navigationTimeoutMs });
	const socket = await websocketPromise;
	invariant(
		new URL(socket.url()).host === baseUrl.host,
		`reload opened an unexpected websocket ${socket.url()}`
	);

	await waitForChatSurface();
	invariant(
		new URL(page.url()).pathname === chatUrl.pathname,
		`reload did not preserve the chat URL ${chatUrl.pathname}; got ${new URL(page.url()).pathname}`
	);
	for (const marker of [initialDefaultMarker, initialAutoMarker]) {
		await page.getByText(marker, { exact: false }).first().waitFor({
			state: 'visible',
			timeout: navigationTimeoutMs
		});
	}
	await sendUiMessage(autoModel, reconnectMarker);
}

async function assertBrowserProviderOutage(modelId) {
	const { networkResponse, networkBody, browserResult } = await browserProxyRequest(
		modelId,
		outageMarker
	);
	invariant(!browserResult.error, browserResult.error);
	invariant(
		networkResponse.status() >= 400 && networkResponse.status() < 600,
		`outage proxy request unexpectedly returned HTTP ${networkResponse.status()}`
	);
	invariant(
		browserResult.status >= 400 && browserResult.status < 600,
		`outage browser fetch unexpectedly returned HTTP ${browserResult.status}`
	);
	invariant(
		networkBody.trim().length > 0 && browserResult.body.trim().length > 0,
		'outage response did not include an actionable provider error body'
	);
}

async function assertUiOutage(modelId) {
	const selectedLabel = await page
		.locator('#model-selector-model-button')
		.getAttribute('aria-label');
	invariant(
		selectedLabel?.includes(modelId),
		`persisted chat did not retain cached model ${modelId}; selector label was ${selectedLabel}`
	);
	const log = page.locator('ul[role="log"]');
	const before = await log.locator('[role="listitem"]').count();
	await page.locator('#chat-input').fill(`This request must fail locally: ${outageMarker}`);
	await page.locator('#send-message-button').click({ timeout: actionTimeoutMs });

	await page.waitForFunction(
		(expectedCount) =>
			document.querySelectorAll('ul[role="log"] [role="listitem"]').length >= expectedCount,
		before + 2,
		{ timeout: responseTimeoutMs }
	);
	const outagePattern =
		/error|issue|server|connection|provider|upstream|not found|502/i;
	const outageItem = log.locator('[role="listitem"]').nth(before + 1);
	await outageItem.filter({ hasText: outagePattern }).waitFor({
		state: 'visible',
		timeout: responseTimeoutMs
	});
	const visibleError = (await outageItem.innerText()).trim();
	invariant(
		outagePattern.test(visibleError),
		'Open WebUI created an outage response without a visible error'
	);
}

async function runInitialPhase() {
	await step('fresh user signup', signupFreshUser);
	await step('direct and AUTO model discovery', assertModelsSelectable);
	await step('document ingest, vector retrieval, and RAG answer', assertRagEndToEnd);
	await step('default browser-visible SSE', () =>
		assertBrowserStreaming(directModel, `${initialDefaultMarker}_SSE`)
	);
	await step('default UI chat', () => sendUiMessage(directModel, initialDefaultMarker));
	await step('AUTO browser-visible SSE', () =>
		assertBrowserStreaming(autoModel, `${initialAutoMarker}_SSE`)
	);
	await step('AUTO UI chat', () => sendUiMessage(autoModel, initialAutoMarker));
	await step('reload, reconnect, and persisted chat', assertReloadReconnectsAndPreservesChat);
}

async function runOutagePhase() {
	await step('existing user sign-in during provider outage', signinExistingUser);
	await step('open persisted chat during provider outage', openPersistedSmokeChat);
	await step('cached model non-success proxy response', () =>
		assertBrowserProviderOutage(autoModel)
	);
	await step('visible UI outage error', () => assertUiOutage(autoModel));
}

async function runRecoveryPhase() {
	await step('existing user sign-in after provider recovery', signinExistingUser);
	await step('open persisted chat after provider recovery', openPersistedSmokeChat);
	await step('open a new chat after provider recovery', async () => {
		await goto('/');
		await waitForChatSurface();
	});
	await step('reselect recovered direct model', () => selectModel(directModel));
	await step('recovered browser-visible SSE', () =>
		assertBrowserStreaming(directModel, `${recoveryMarker}_SSE`)
	);
	await step('recovered UI chat', () => sendUiMessage(directModel, recoveryMarker));
	await step('RAG retrieval and answer after gateway restart', assertRagAfterGatewayRestart);
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
	for (const identity of [
		'Final answer attribution',
		'publisher',
		'tier1',
		'tier2',
		'qwen3.8-27b',
		'deepseek-v4-flash-0731-thinking'
	]) {
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

async function runTieredPhase() {
	await step('open no-auth tiered chat', async () => {
		await goto('/');
		await waitForChatSurface();
	});
	await step('single product model inventory', assertTieredProductInventory);
	await step('folded attributed reasoning and separate final answer', assertTieredReasoningUi);
}

async function main() {
	invariant(
		PHASES.has(phase),
		`WEBUI_SMOKE_PHASE must be initial, outage, recovery, or tiered; got ${phase}`
	);
	invariant(
		['http:', 'https:'].includes(baseUrl.protocol),
		`WEBUI_SMOKE_BASE_URL must be HTTP(S), got ${baseUrl.href}`
	);
	invariant(baseUrl.pathname === '/', `WEBUI_SMOKE_BASE_URL must not contain a path: ${baseUrl.href}`);
	if (phase === 'tiered') {
		invariant(
			productModel === 'kairyu-auto-max',
			`EXPECTED_PRODUCT_MODEL must be kairyu-auto-max, got ${productModel}`
		);
	} else {
		invariant(directModel === 'default', `EXPECTED_DIRECT_MODEL must be default, got ${directModel}`);
		invariant(autoModel === 'kairyu-auto', `EXPECTED_AUTO_MODEL must be kairyu-auto, got ${autoModel}`);
	}

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

	if (phase === 'initial') {
		await runInitialPhase();
	} else if (phase === 'outage') {
		await runOutagePhase();
	} else if (phase === 'recovery') {
		await runRecoveryPhase();
	} else {
		await runTieredPhase();
	}

	console.log(`WEBUI BROWSER SMOKE PASS: ${phase}`);
}

try {
	await bounded(main(), phaseTimeoutMs, `${phase} browser smoke phase`);
} catch (error) {
	const detail = error instanceof Error ? error.stack ?? error.message : String(error);
	console.error(`WEBUI BROWSER SMOKE FAIL [phase=${phase}] [step=${currentStep}]\n${detail}`);
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
