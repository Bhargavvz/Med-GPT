import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, X, Loader2, ArrowRight, Sparkles, ImageIcon } from 'lucide-react'
import './Analyze.css'

const API = import.meta.env.VITE_API_URL || ''

const suggestions = [
    'What abnormality is visible in this image?',
    'What organ is shown?',
    'Is there any sign of disease?',
    'What imaging modality was used?',
    'What is the diagnosis?',
]

export default function Analyze() {
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [question, setQuestion] = useState('')
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [dragOver, setDragOver] = useState(false)
    const inputRef = useRef(null)
    const resultRef = useRef(null)

    const handleFile = useCallback((f) => {
        if (!f || !f.type.startsWith('image/')) return setError('Please upload a valid image.')
        if (f.size > 50 * 1024 * 1024) return setError('Image too large (max 50MB)')
        setFile(f)
        setError(null)
        const reader = new FileReader()
        reader.onload = (e) => setPreview(e.target.result)
        reader.readAsDataURL(f)
    }, [])

    const removeFile = () => { setFile(null); setPreview(null); if (inputRef.current) inputRef.current.value = '' }

    const handleDrop = (e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]) }

    const analyze = async () => {
        if (!file || !question.trim()) return
        setLoading(true); setError(null); setResult(null)

        try {
            const form = new FormData()
            form.append('image', file)
            form.append('question', question.trim())
            form.append('generate_heatmap', 'true')

            const res = await fetch(`${API}/api/predict`, { method: 'POST', body: form })
            if (!res.ok) {
                const err = await res.json().catch(() => ({}))
                throw new Error(err.detail || `Server error: ${res.status}`)
            }
            const data = await res.json()
            setResult(data)
            setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
        } catch (err) {
            setError(err.message || 'Something went wrong.')
        } finally {
            setLoading(false)
        }
    }

    const handleKey = (e) => {
        if (e.key === 'Enter' && !e.shiftKey && file && question.trim()) { e.preventDefault(); analyze() }
    }

    return (
        <div className="analyze-page container">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
                {/* Header */}
                <div className="analyze-header">
                    <h1><Sparkles size={28} className="gradient-icon" /> Medical Image Analysis</h1>
                    <p>Upload a medical image and ask any question about it</p>
                </div>

                {/* Upload + Question */}
                <div className="analyze-input glass-card">
                    {/* Image Upload */}
                    <div
                        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${preview ? 'has-image' : ''}`}
                        onClick={() => !preview && inputRef.current?.click()}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                    >
                        {preview ? (
                            <div className="upload-preview">
                                <img src={preview} alt="Uploaded" />
                                <button className="btn-remove" onClick={(e) => { e.stopPropagation(); removeFile() }}>
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <div className="upload-content">
                                <div className="upload-icon-wrap">
                                    <Upload size={32} />
                                </div>
                                <p className="upload-text">Drop medical image here or <span className="upload-link">browse</span></p>
                                <p className="upload-hint">X-ray, CT, MRI, Ultrasound, or Pathology slide</p>
                                <p className="upload-formats">JPEG, PNG — max 50MB</p>
                            </div>
                        )}
                        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])} />
                    </div>

                    {/* Question */}
                    <div className="question-group">
                        <label className="input-label">Your Question</label>
                        <div className="question-row">
                            <textarea
                                className="question-input"
                                placeholder="e.g., What abnormality is visible in this chest X-ray?"
                                rows={2}
                                value={question}
                                onChange={(e) => setQuestion(e.target.value)}
                                onKeyDown={handleKey}
                            />
                            <button
                                className="btn-primary btn-analyze"
                                disabled={!file || !question.trim() || loading}
                                onClick={analyze}
                            >
                                {loading ? <Loader2 size={18} className="spin" /> : <><span>Analyze</span><ArrowRight size={16} /></>}
                            </button>
                        </div>
                    </div>

                    {/* Suggestions */}
                    <div className="suggestions">
                        <span className="suggestions-label">Try:</span>
                        {suggestions.map(s => (
                            <button key={s} className="chip" onClick={() => setQuestion(s)}>
                                {s.length > 30 ? s.slice(0, 28) + '…' : s}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Error */}
                <AnimatePresence>
                    {error && (
                        <motion.div className="error-banner" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                            ⚠️ {error}
                            <button onClick={() => setError(null)}><X size={14} /></button>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Results */}
                <AnimatePresence>
                    {result && (
                        <motion.div
                            ref={resultRef}
                            className="results"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            {/* Answer */}
                            <div className="result-card glass-card">
                                <div className="result-header">
                                    <h3>💡 Answer</h3>
                                    <span className="result-time">{result.processing_time}s</span>
                                </div>
                                <div className="answer-body">
                                    <p className="answer-text">{result.answer}</p>
                                </div>
                            </div>

                            {/* Heatmap */}
                            {result.heatmap && (
                                <div className="result-card glass-card">
                                    <div className="result-header">
                                        <h3>🔍 Visual Explanation (Grad-CAM)</h3>
                                    </div>
                                    <div className="image-comparison">
                                        <div className="comparison-item">
                                            <span className="comparison-label">Original</span>
                                            <img src={`data:image/png;base64,${result.original_image}`} alt="Original" />
                                        </div>
                                        <div className="comparison-divider">→</div>
                                        <div className="comparison-item">
                                            <span className="comparison-label">Model Focus</span>
                                            <img src={`data:image/png;base64,${result.heatmap}`} alt="Heatmap" />
                                        </div>
                                    </div>
                                    <p className="visual-caption">
                                        Warm colors (red/yellow) indicate regions the model focused on to generate its answer.
                                    </p>
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </div>
    )
}
