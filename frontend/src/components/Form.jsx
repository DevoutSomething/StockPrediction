import React from "react";
import "./Styles/form.css";

export default function Form({ state, setState, text, defaultText, id }) {
  const inputClass = state ? "valid" : "empty";
  const labelClass = state ? "valid" : "empty";

  const handleInputChange = (e) => {
    const value = e.target.value;

    if (!isNaN(value) && value !== "" && parseFloat(e.target.value) > 0) {
      setState(value);
    } else {
      setState("");
    }
  };

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
        onChange={handleInputChange}
        className={inputClass}
      />
    </div>
  );
}
