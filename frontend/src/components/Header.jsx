import Logo from "./Logo";
import "./Styles/Header.css";
import { Link } from "react-router-dom";

const Header = ({ backButton = false }) => {
  return (
    <div className="header">
      <div className="header-content">
        <Logo className="logo" route="/" />
        {backButton && (
          <Logo logoSrc="/images/retry.png" className="back" route="/params" />
        )}
        <h1>Stock Predictor</h1>
        <nav className="main-nav"></nav>
      </div>
    </div>
  );
};

export default Header;
