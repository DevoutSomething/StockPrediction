import { Link } from "react-router-dom"

const Logo = ({ logoSrc = "/images/logo.png", className, route }) => {
  return (
    <div className="logo-container">
      <Link to={route}>
        <img src={logoSrc || "/placeholder.svg"} alt="Stock Predictor" className={className} />
      </Link>
    </div>
  )
}

export default Logo

