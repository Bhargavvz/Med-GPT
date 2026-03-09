import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Brain, Upload, BarChart3, Eye, Zap, Shield, ArrowRight } from 'lucide-react'
import './Home.css'

const fadeUp = {
    hidden: { opacity: 0, y: 30 },
    visible: (i) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.6, ease: [0.16, 1, 0.3, 1] } })
}

const features = [
    { icon: <Brain size={24} />, title: 'Vision-Language AI', desc: 'Powered by Qwen3-VL-8B fine-tuned on 150K+ medical VQA samples' },
    { icon: <Eye size={24} />, title: 'Grad-CAM Heatmaps', desc: 'See exactly where the model focuses to generate its answer' },
    { icon: <BarChart3 size={24} />, title: '78.5% Accuracy', desc: '87.1% on yes/no questions, 82.8% BLEU-1 score on open-ended' },
    { icon: <Zap size={24} />, title: 'Real-time Inference', desc: 'Get answers in seconds with GPU-accelerated processing' },
    { icon: <Upload size={24} />, title: 'Multi-Modality', desc: 'Supports X-ray, CT, MRI, Ultrasound, and Pathology slides' },
    { icon: <Shield size={24} />, title: 'Explainable AI', desc: 'Transparent reasoning with attention visualization and rationale' },
]

const stats = [
    { value: '78.5%', label: 'Accuracy' },
    { value: '82.8%', label: 'BLEU-1' },
    { value: '150K+', label: 'Training Samples' },
    { value: '8B', label: 'Parameters' },
]

export default function Home() {
    const navigate = useNavigate()

    return (
        <div className="home-page">
            {/* Hero */}
            <section className="hero container">
                <motion.div className="hero-content" initial="hidden" animate="visible">
                    <motion.span className="hero-badge" variants={fadeUp} custom={0}>
                        🏥 Medical Visual Question Answering
                    </motion.span>
                    <motion.h1 className="hero-title" variants={fadeUp} custom={1}>
                        AI-Powered <span className="gradient-text">Medical Image</span> Analysis
                    </motion.h1>
                    <motion.p className="hero-subtitle" variants={fadeUp} custom={2}>
                        Upload any medical image — X-ray, CT, MRI, or pathology slide — and ask questions
                        in natural language. Get expert-level answers with visual explanations.
                    </motion.p>
                    <motion.div className="hero-actions" variants={fadeUp} custom={3}>
                        <button className="btn-primary hero-cta" onClick={() => navigate('/analyze')}>
                            Start Analyzing <ArrowRight size={18} />
                        </button>
                        <button className="btn-secondary" onClick={() => navigate('/about')}>
                            Learn More
                        </button>
                    </motion.div>
                </motion.div>

                {/* Stats */}
                <motion.div className="hero-stats" initial="hidden" animate="visible">
                    {stats.map((s, i) => (
                        <motion.div key={s.label} className="stat-card glass-card" variants={fadeUp} custom={i + 4}>
                            <span className="stat-value gradient-text">{s.value}</span>
                            <span className="stat-label">{s.label}</span>
                        </motion.div>
                    ))}
                </motion.div>
            </section>

            {/* Features */}
            <section className="features container">
                <motion.h2 className="section-heading" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
                    <span className="gradient-text">Key Features</span>
                </motion.h2>
                <div className="features-grid">
                    {features.map((f, i) => (
                        <motion.div
                            key={f.title}
                            className="feature-card glass-card"
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: i * 0.08, duration: 0.5 }}
                        >
                            <div className="feature-icon">{f.icon}</div>
                            <h3 className="feature-title">{f.title}</h3>
                            <p className="feature-desc">{f.desc}</p>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* CTA */}
            <section className="cta-section container">
                <motion.div
                    className="cta-card glass-card"
                    initial={{ opacity: 0, scale: 0.95 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                >
                    <h2>Ready to try MedGPT?</h2>
                    <p>Upload a medical image and get AI-powered analysis in seconds.</p>
                    <button className="btn-primary" onClick={() => navigate('/analyze')}>
                        Get Started <ArrowRight size={18} />
                    </button>
                </motion.div>
            </section>
        </div>
    )
}
