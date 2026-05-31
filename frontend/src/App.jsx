import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Login from './pages/Login'
import Chat from './pages/Chat'
import Dashboard from './pages/Dashboard'
import Analytics from './pages/Analytics'
import FraudTool from './pages/FraudTool'
import Layout from './components/Layout'
import { getToken } from './lib/api'

function Protected({ children }) {
  return getToken() ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />

        {/* Protected routes share the Layout (sidebar + outlet) */}
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

        {/* Land on /chat by default — Protected sends to /login if not signed in */}
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </BrowserRouter>
  )
}