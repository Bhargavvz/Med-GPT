import { Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Analyze from './pages/Analyze'
import Dashboard from './pages/Dashboard'
import About from './pages/About'

function App() {
    return (
        <>
            {/* Animated background */}
            <div className="bg-grid" />
            <div className="bg-glow bg-glow-1" />
            <div className="bg-glow bg-glow-2" />
            <div className="bg-glow bg-glow-3" />

            <Navbar />
            <div className="page-wrapper">
                <AnimatePresence mode="wait">
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/analyze" element={<Analyze />} />
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/about" element={<About />} />
                    </Routes>
                </AnimatePresence>
            </div>
            <Footer />
        </>
    )
}

export default App
