import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, TrendingDown, Target, Activity } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts'
import './Dashboard.css'

const API = import.meta.env.VITE_API_URL || ''

export default function Dashboard() {
    const [metrics, setMetrics] = useState(null)
    const [trainHistory, setTrainHistory] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function load() {
            try {
                const [mRes, tRes] = await Promise.all([
                    fetch(`${API}/api/metrics`).catch(() => null),
                    fetch(`${API}/api/training-history`).catch(() => null),
                ])
                if (mRes?.ok) setMetrics(await mRes.json())
                if (tRes?.ok) setTrainHistory(await tRes.json())
            } catch (e) { console.error(e) }
            setLoading(false)
        }
        load()
    }, [])

    // Format training curves
    const trainLoss = trainHistory?.log_history?.filter(l => l.loss)?.map(l => ({ step: l.step, loss: l.loss })) || []
    const evalLoss = trainHistory?.log_history?.filter(l => l.eval_loss)?.map(l => ({ step: l.step, eval_loss: l.eval_loss })) || []

    // Accuracy data
    const accData = metrics ? [
        { name: 'Overall', accuracy: (metrics.accuracy * 100).toFixed(1) },
        { name: 'Yes/No', accuracy: ((metrics.accuracy_closed || 0) * 100).toFixed(1) },
        { name: 'Open-ended', accuracy: ((metrics.accuracy_open || 0) * 100).toFixed(1) },
    ] : []

    // Radar data
    const radarData = metrics ? [
        { metric: 'Accuracy', value: metrics.accuracy * 100 },
        { metric: 'BLEU-1', value: (metrics.bleu1 || 0) * 100 },
        { metric: 'ROUGE-L', value: (metrics.rouge_l || 0) * 100 },
        { metric: 'Token F1', value: (metrics.token_f1 || 0) * 100 },
    ] : []

    const summaryCards = metrics ? [
        { icon: <Target size={20} />, label: 'Overall Accuracy', value: `${(metrics.accuracy * 100).toFixed(1)}%`, color: '#10b981' },
        { icon: <BarChart3 size={20} />, label: 'BLEU-1 Score', value: `${((metrics.bleu1 || 0) * 100).toFixed(1)}%`, color: '#6366f1' },
        { icon: <Activity size={20} />, label: 'ROUGE-L Score', value: `${((metrics.rouge_l || 0) * 100).toFixed(1)}%`, color: '#06b6d4' },
        { icon: <TrendingDown size={20} />, label: 'Test Samples', value: `${metrics.n_samples || 0}`, color: '#f59e0b' },
    ] : []

    if (loading) {
        return (
            <div className="dashboard-page container">
                <div className="dashboard-loading">
                    <div className="loader" />
                    <p>Loading metrics...</p>
                </div>
            </div>
        )
    }

    if (!metrics && !trainHistory) {
        return (
            <div className="dashboard-page container">
                <div className="dashboard-empty glass-card">
                    <BarChart3 size={48} />
                    <h2>No Metrics Available</h2>
                    <p>Run evaluation first to see the dashboard:</p>
                    <code>python training/evaluate.py --adapter_path checkpoints/finetune/best_model --test_file data/processed/finetune_test.json --output_file results/eval_results.json</code>
                </div>
            </div>
        )
    }

    return (
        <div className="dashboard-page container">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="dashboard-header">
                    <h1><BarChart3 size={28} className="gradient-icon" /> Model Dashboard</h1>
                    <p>Training metrics and evaluation results</p>
                </div>

                {/* Summary Cards */}
                {metrics && (
                    <div className="summary-grid">
                        {summaryCards.map((c, i) => (
                            <motion.div
                                key={c.label}
                                className="summary-card glass-card"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.1 }}
                            >
                                <div className="summary-icon" style={{ color: c.color, background: `${c.color}15` }}>{c.icon}</div>
                                <div className="summary-info">
                                    <span className="summary-value">{c.value}</span>
                                    <span className="summary-label">{c.label}</span>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                )}

                <div className="charts-grid">
                    {/* Accuracy Breakdown */}
                    {metrics && (
                        <div className="chart-card glass-card">
                            <h3>Accuracy by Question Type</h3>
                            <ResponsiveContainer width="100%" height={280}>
                                <BarChart data={accData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                                    <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} />
                                    <Tooltip
                                        contentStyle={{ background: '#1a2035', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, color: '#f1f5f9' }}
                                        formatter={(v) => [`${v}%`, 'Accuracy']}
                                    />
                                    <Bar dataKey="accuracy" fill="url(#barGrad)" radius={[6, 6, 0, 0]} />
                                    <defs>
                                        <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#6366f1" />
                                            <stop offset="100%" stopColor="#06b6d4" />
                                        </linearGradient>
                                    </defs>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Radar Chart */}
                    {metrics && (
                        <div className="chart-card glass-card">
                            <h3>Performance Overview</h3>
                            <ResponsiveContainer width="100%" height={280}>
                                <RadarChart data={radarData}>
                                    <PolarGrid stroke="rgba(255,255,255,0.08)" />
                                    <PolarAngleAxis dataKey="metric" stroke="#94a3b8" fontSize={12} />
                                    <PolarRadiusAxis domain={[0, 100]} tick={false} />
                                    <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} strokeWidth={2} />
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Training Loss */}
                    {trainLoss.length > 0 && (
                        <div className="chart-card glass-card chart-wide">
                            <h3>Training Loss</h3>
                            <ResponsiveContainer width="100%" height={280}>
                                <LineChart data={trainLoss}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="step" stroke="#64748b" fontSize={11} />
                                    <YAxis stroke="#64748b" fontSize={11} />
                                    <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, color: '#f1f5f9' }} />
                                    <Line type="monotone" dataKey="loss" stroke="#6366f1" strokeWidth={1.5} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Eval Loss */}
                    {evalLoss.length > 0 && (
                        <div className="chart-card glass-card chart-wide">
                            <h3>Validation Loss</h3>
                            <ResponsiveContainer width="100%" height={280}>
                                <LineChart data={evalLoss}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="step" stroke="#64748b" fontSize={11} />
                                    <YAxis stroke="#64748b" fontSize={11} />
                                    <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, color: '#f1f5f9' }} />
                                    <Line type="monotone" dataKey="eval_loss" stroke="#f44336" strokeWidth={2} dot={{ r: 4, fill: '#f44336' }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </div>
            </motion.div>
        </div>
    )
}
