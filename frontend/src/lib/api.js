const BASE = import.meta.env.VITE_API_URL || ''

export const getToken = () => sessionStorage.getItem('insurai_token')
export const getUser  = () => sessionStorage.getItem('insurai_user') || 'User'

export function logout() {
  sessionStorage.removeItem('insurai_token')
  sessionStorage.removeItem('insurai_user')
  fetch(BASE + '/logout', { method: 'POST' }).finally(() => {
    window.location.href = '/login'
  })
}

export async function authFetch(path, options = {}) {
  const token = getToken()
  const res = await fetch(BASE + path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  })
  if (res.status === 401 && token) logout()
  return res
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