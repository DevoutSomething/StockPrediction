import React from "react";
import { Link } from "react-router-dom";
import "./Styles/paramsButton.css"; // Import the CSS for styling

export default function ParamsBtn({ route }) {
  return (
    <nav>
      <Link to={route} className="params-button">
        <p>Try it</p>
      </Link>
    </nav>
  );
}
