import Logo from "./Logo";
import "./Styles/Header.css";
import { Link } from "react-router-dom";
const Header = () => {
  return (
    <div className="header">
      <Logo className="logo" route="/" />
      <h1>Stock Predictor</h1>
    </div>
  );
};

export default Header;
