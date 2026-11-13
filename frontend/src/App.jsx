import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Login from './pages/Login'
import DemoPage from './pages/DemoPage'
import Chat from './pages/Chat'
import Dashboard from './pages/Dashboard'
import Analytics from './pages/Analytics'
import FraudTool from './pages/FraudTool'
import Layout from './components/Layout'
import { getToken } from './lib/api'

function isTokenValid(token) {
  try {
    const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')))
    return !payload.exp || payload.exp * 1000 > Date.now()
  } catch {
    return false
  }
}

function Protected({ children }) {
  const token = getToken()
  if (!token || !isTokenValid(token)) {
    sessionStorage.removeItem('insurai_token')
    return <Navigate to="/demo" replace />
  }
  return children
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/login" element={<Login />} />

        <Route
          element={
            <Protected>
              <Layout />
            </Protected>
          }
        >
          <Route path="/chat" element={<Chat />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/fraud" element={<FraudTool />} />
          <Route path="/fraud/:threadId" element={<FraudTool />} />
        </Route>

        {/* Startup page: signed-out visitors land on /demo, signed-in ones on /chat */}
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </BrowserRouter>
  )
}