import Logo from "./Logo"
import "./Styles/Header.css"

const Header = ({ backButton = false }) => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo-container">
          <Logo className="logo" route="/" />
          <div className="market-indicator">
            <span className="indicator up"></span>
            <span className="indicator down"></span>
          </div>
        </div>
        <h1 className="header-title">
          Stock<span>Predictor</span>
        </h1>
        {backButton && (
          <div className="back-button-container">
            <Logo logoSrc="/images/retry.png" className="back" route="/params" />
          </div>
        )}
      </div>
    </header>
  )
}

export default Header

