import Logo from "./Logo";
import "./Styles/Header.css";
import { Link } from "react-router-dom";

const Header = ({ backButton = false }) => {
  return (
    <div className="header">
      <div className="header-content">
        <div className="logo-container">
          <Logo className="logo" route="/" />
        </div>
        <h1>Stock Predictor</h1>
        {backButton && (
          <Logo logoSrc="/images/retry.png" className="back" route="/params" />
        )}
      </div>
    </div>
  );
};

export default Header;
