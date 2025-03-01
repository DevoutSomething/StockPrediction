import React from "react";
import "./Styles/form.css";

export default function Form({ state, setState, text, defaultText, id }) {
  const inputClass = state ? "valid" : "empty";
  const labelClass = state ? "valid" : "empty";
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
        onChange={(e) =>
          e.target.val > 0 ? setState(e.target.value) : setState("")
        }
        className={inputClass}
      />
    </div>
  );
}
