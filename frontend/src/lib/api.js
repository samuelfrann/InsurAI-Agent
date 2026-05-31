const BASE = import.meta.env.VITE_API_URL || ''

export const getToken = () => localStorage.getItem('insurai_token')
export const getUser  = () => localStorage.getItem('insurai_user') || 'User'

export function authFetch(path, options = {}) {
  const token = getToken()
  return fetch(BASE + path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  })
}

export function logout() {
  localStorage.removeItem('insurai_token')
  localStorage.removeItem('insurai_user')
  fetch(BASE + '/logout', { method: 'POST' }).finally(() => {
    window.location.href = '/login'
  })
}

export function newThreadId() {
  return 'insurai_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7)
}

export async function getFxRate() {
  const res = await authFetch('/config/fx-rate')
  const data = await res.json()
  return data.ngn_per_usd
}

export async function assessClaim(claimData) {
  const res = await authFetch('/fraud/assess', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(claimData),
  })
  return res.json()
}

export async function createFraudSession() {
  const res = await authFetch('/fraud/sessions', { method: 'POST' })
  return res.json()
}

export async function saveFraudResult(threadId, payload) {
  const res = await authFetch(`/fraud/sessions/${threadId}/result`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return res.json()
}