import { Heart } from 'lucide-react'
import './Footer.css'

export default function Footer() {
    return (
        <footer className="footer">
            <div className="container footer-inner">
                <p className="footer-text">
                    MedGPT — Research & Educational Use Only. Not for clinical diagnosis.
                </p>
                <p className="footer-credit">
                    Built with <Heart size={12} className="heart" /> by Bhargav
                </p>
            </div>
        </footer>
    )
}
