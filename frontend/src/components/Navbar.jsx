import { useState, useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Activity, Menu, X } from 'lucide-react'
import './Navbar.css'

const API = import.meta.env.VITE_API_URL || ''

export default function Navbar() {
    const [modelStatus, setModelStatus] = useState('checking')
    const [mobileOpen, setMobileOpen] = useState(false)
    const location = useLocation()

    useEffect(() => { setMobileOpen(false) }, [location])

    useEffect(() => {
        async function check() {
            try {
                const res = await fetch(`${API}/api/health`)
                const data = await res.json()
                setModelStatus(data.model_loaded ? 'online' : 'loading')
                if (!data.model_loaded) setTimeout(check, 10000)
            } catch {
                setModelStatus('offline')
                setTimeout(check, 15000)
            }
        }
        check()
    }, [])

    const links = [
        { to: '/', label: 'Home' },
        { to: '/analyze', label: 'Analyze' },
        { to: '/dashboard', label: 'Dashboard' },
        { to: '/about', label: 'About' },
    ]

    const statusLabel = {
        checking: 'Checking model...',
        online: 'Model ready',
        loading: 'Model loading...',
        offline: 'Server offline',
    }

    return (
        <header className="navbar">
            <div className="navbar-inner container">
                <NavLink to="/" className="navbar-logo">
                    <div className="logo-icon">
                        <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                            <rect x="2" y="10" width="24" height="8" rx="2" fill="url(#navGrad)" opacity="0.8" />
                            <rect x="10" y="2" width="8" height="24" rx="2" fill="url(#navGrad)" />
                            <defs>
                                <linearGradient id="navGrad" x1="0" y1="0" x2="28" y2="28">
                                    <stop offset="0%" stopColor="#6366f1" />
                                    <stop offset="100%" stopColor="#06b6d4" />
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>
                    <span className="logo-text gradient-text">MedGPT</span>
                    <span className="logo-badge">AI</span>
                </NavLink>

                <nav className={`navbar-links ${mobileOpen ? 'open' : ''}`}>
                    {links.map(l => (
                        <NavLink
                            key={l.to}
                            to={l.to}
                            end={l.to === '/'}
                            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                        >
                            {l.label}
                        </NavLink>
                    ))}
                </nav>

                <div className="navbar-right">
                    <div className={`status-badge ${modelStatus}`}>
                        <span className={`status-dot ${modelStatus}`} />
                        <span className="status-text">{statusLabel[modelStatus]}</span>
                    </div>
                    <button className="mobile-toggle" onClick={() => setMobileOpen(!mobileOpen)}>
                        {mobileOpen ? <X size={20} /> : <Menu size={20} />}
                    </button>
                </div>
            </div>
        </header>
    )
}
