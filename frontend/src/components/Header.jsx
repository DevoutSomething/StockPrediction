import Logo from "./Logo";
import "./Styles/Header.css";

const Header = ({ backButton = false }) => {
  return (
    <header className="header">
      <div className="header-content">
        {backButton && (
          <div className="back-button-container">
            <Logo
              logoSrc="/images/retry.png"
              className="back"
              route="/params"
            />
          </div>
        )}

        <div className="logo-container">
          <Logo className="logo" route="/" />
        </div>

        <div className="market-indicator">
          <span className="indicator up"></span>
          <span className="indicator down"></span>
        </div>
      </div>
    </header>
  );
};

export default Header;
