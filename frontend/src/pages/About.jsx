import { motion } from 'framer-motion'
import { Brain, Layers, Database, Cpu, GitBranch, BookOpen } from 'lucide-react'
import './About.css'

const steps = [
    { icon: <Database size={20} />, title: 'Data Collection', desc: '150K+ medical VQA samples from PMC-VQA, VQA-RAD, SLAKE, and PathVQA datasets' },
    { icon: <Layers size={20} />, title: 'LoRA Fine-tuning', desc: 'Only 2.3% of parameters trained (210M out of 9B) using Low-Rank Adaptation' },
    { icon: <Cpu size={20} />, title: 'Vision-Language Model', desc: 'Qwen3-VL-8B base model processes both images and text in a unified architecture' },
    { icon: <Brain size={20} />, title: 'Inference + Explainability', desc: 'Generates answers with Grad-CAM heatmaps showing model attention regions' },
]

const techStack = [
    { name: 'Qwen3-VL-8B', desc: 'Vision-Language Model' },
    { name: 'LoRA', desc: 'Parameter-Efficient Fine-Tuning' },
    { name: 'PyTorch', desc: 'Deep Learning Framework' },
    { name: 'HuggingFace', desc: 'Model Hub & Transformers' },
    { name: 'FastAPI', desc: 'Backend Server' },
    { name: 'React + Vite', desc: 'Frontend UI' },
    { name: 'Grad-CAM', desc: 'Visual Explainability' },
    { name: 'NVIDIA H200', desc: 'GPU Training & Inference' },
]

const datasets = [
    { name: 'PMC-VQA', samples: '~140K', type: 'Pre-training' },
    { name: 'VQA-RAD', samples: '~3.5K', type: 'Fine-tuning' },
    { name: 'SLAKE', samples: '~14K', type: 'Fine-tuning' },
    { name: 'PathVQA', samples: '~32K', type: 'Fine-tuning' },
]

export default function About() {
    return (
        <div className="about-page container">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="about-header">
                    <h1><BookOpen size={28} className="gradient-icon" /> About MedGPT</h1>
                    <p>An AI system for medical visual question answering</p>
                </div>

                {/* Intro */}
                <div className="about-intro glass-card">
                    <h2>What is MedGPT?</h2>
                    <p>
                        MedGPT is a vision-language AI model fine-tuned specifically for medical image analysis.
                        It can understand medical images (X-rays, CT scans, MRIs, pathology slides) and answer
                        questions about them in natural language. The model achieves <strong>78.5% accuracy</strong> on
                        medical VQA benchmarks, with <strong>87.1%</strong> on yes/no questions.
                    </p>
                    <p>
                        Unlike generic AI models, MedGPT is trained on carefully curated medical datasets and
                        provides visual explanations (Grad-CAM heatmaps) showing where the model is looking
                        to generate its answer — making it transparent and interpretable.
                    </p>
                </div>

                {/* How it Works */}
                <div className="how-it-works">
                    <h2 className="section-title gradient-text">How It Works</h2>
                    <div className="steps-grid">
                        {steps.map((s, i) => (
                            <motion.div
                                key={s.title}
                                className="step-card glass-card"
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: i * 0.1 }}
                            >
                                <div className="step-number">{i + 1}</div>
                                <div className="step-icon">{s.icon}</div>
                                <h3>{s.title}</h3>
                                <p>{s.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Architecture */}
                <div className="architecture glass-card">
                    <h2>Architecture</h2>
                    <div className="arch-diagram">
                        <div className="arch-block">
                            <span className="arch-label">Input</span>
                            <span>Medical Image + Question</span>
                        </div>
                        <div className="arch-arrow">→</div>
                        <div className="arch-block highlight">
                            <span className="arch-label">Model</span>
                            <span>Qwen3-VL-8B + LoRA</span>
                        </div>
                        <div className="arch-arrow">→</div>
                        <div className="arch-block">
                            <span className="arch-label">Output</span>
                            <span>Answer + Heatmap</span>
                        </div>
                    </div>
                </div>

                {/* Datasets */}
                <div className="datasets-section glass-card">
                    <h2>Training Datasets</h2>
                    <div className="datasets-table">
                        <div className="table-header">
                            <span>Dataset</span><span>Samples</span><span>Stage</span>
                        </div>
                        {datasets.map(d => (
                            <div key={d.name} className="table-row">
                                <span className="dataset-name">{d.name}</span>
                                <span>{d.samples}</span>
                                <span className={`stage-badge ${d.type === 'Pre-training' ? 'pretrain' : 'finetune'}`}>{d.type}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Tech Stack */}
                <div className="tech-section">
                    <h2 className="section-title gradient-text">Technology Stack</h2>
                    <div className="tech-grid">
                        {techStack.map((t, i) => (
                            <motion.div
                                key={t.name}
                                className="tech-chip glass-card"
                                initial={{ opacity: 0, scale: 0.9 }}
                                whileInView={{ opacity: 1, scale: 1 }}
                                viewport={{ once: true }}
                                transition={{ delay: i * 0.05 }}
                            >
                                <span className="tech-name">{t.name}</span>
                                <span className="tech-desc">{t.desc}</span>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Disclaimer */}
                <div className="disclaimer glass-card">
                    <h3>⚠️ Disclaimer</h3>
                    <p>
                        MedGPT is a research and educational tool. It is <strong>NOT</strong> intended for
                        clinical diagnosis or medical decision-making. Always consult qualified healthcare
                        professionals for medical advice.
                    </p>
                </div>
            </motion.div>
        </div>
    )
}
