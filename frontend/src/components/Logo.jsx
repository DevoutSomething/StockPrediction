import { Link } from "react-router-dom";
import { GlobalContext } from "../context";
import { useContext } from "react";
const Logo = ({ logoSrc = "/images/logo.png", className, route }) => {
  const { setPayment, setTime, setProfit } = useContext(GlobalContext);
  const handleClick = () => {
    setPayment("");
    setTime("");
    setProfit("");
  };
  return (
    <div className="logo-container">
      <Link to={route}>
        <img
          src={logoSrc || "/placeholder.svg"}
          alt="Stock Predictor"
          className={className}
          onClick={handleClick}
        />
      </Link>
    </div>
  );
};

export default Logo;
