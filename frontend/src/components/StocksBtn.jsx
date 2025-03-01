import React from "react";
import { Link } from "react-router-dom";
import "./Styles/stocksButton.css";

export default function StocksBtn({ route }) {
  return (
    <nav>
      <Link to={route} className="stocks-button">
        <p>Get your stocks!</p>
      </Link>
    </nav>
  );
}
