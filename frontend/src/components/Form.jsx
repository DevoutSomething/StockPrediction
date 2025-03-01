import React from "react";
import "./Styles/form.css";

export default function Form({ state, setState, text, defaultText, id }) {
  // Determine if the value is falsy or truthy
  const inputClass = state ? "valid" : "empty";
  const labelClass = state ? "valid" : "empty"; // Apply the same logic to the label

  return (
    <div className="input-group">
      <label htmlFor={id} className={labelClass}>
        {text}
      </label>
      <input
        type="number"
        id={id}
        name={id}
        placeholder={defaultText}
        value={state}
        onChange={(e) => setState(e.target.value)}
        className={inputClass} // Apply class based on input value
      />
    </div>
  );
}
