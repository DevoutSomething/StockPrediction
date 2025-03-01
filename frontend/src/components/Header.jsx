import Logo from "./Logo";
import "./Styles/Header.css";
import { Link } from "react-router-dom";

const Header = () => {
  return (
    <div className="header">
      <div className="header-content">
        <Logo className="logo" route="/" />
        <h1>Stock Predictor</h1>
        <nav className="main-nav">
          <Link to="/" className="nav-link">
          </Link>
          <Link to="/params" className="nav-link">
          </Link>
          <Link to="/stocks" className="nav-link">
            
          </Link>
        </nav>
      </div>
    </div>
  );
};

export default Header;
