import { Link } from "react-router-dom";

const Logo = ({ logoSrc = "/images/logo.png", className, route }) => {
  return (
    <nav>
      <Link to={route}>
        <img src={logoSrc} alt="logo" className={className} />
      </Link>
    </nav>
  );
};

export default Logo;
