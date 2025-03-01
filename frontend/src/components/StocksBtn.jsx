import React from "react";
import { Link } from "react-router-dom";
import "./Styles/stocksButton.css"; // Import the CSS for styling

export default function StocksBtn({ route, isValid }) {
  const buttonClass = isValid ? "valid" : "invalid";

  return (
    <nav>
      <Link to={route} className={`stocks-button ${buttonClass}`}>
        Try it
      </Link>
    </nav>
  );
}
