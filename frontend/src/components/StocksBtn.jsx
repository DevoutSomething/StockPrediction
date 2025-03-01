import React from "react";
import { Link } from "react-router-dom";
import "./Styles/stocksButton.css"; // Import the CSS for styling

export default function StocksBtn({ route, isValid }) {
  // Set button class based on validity (if applicable)
  const buttonClass = isValid ? "valid" : "invalid"; // Use "valid" or "invalid" based on your condition

  return (
    <nav>
      <Link to={route} className={`stocks-button ${buttonClass}`}>
        Try it
      </Link>
    </nav>
  );
}
