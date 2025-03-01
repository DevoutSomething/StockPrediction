import React from "react";
import "./Styles/form.css";
import { useRef, useEffect } from "react";
export default function Form({
  state,
  setState,
  text,
  defaultText,
  id,
  isFirst = false,
}) {
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

  // Create a ref for the first form element
  const firstFormElementRef = useRef(null);

  // Focus the first form element when the component mounts
  useEffect(() => {
    if (firstFormElementRef.current && isFirst) {
      firstFormElementRef.current.focus();
    }
  }, [isFirst]);

  return (
    <div className="input-group">
      <label htmlFor={id} className={labelClass}>
        {text}
      </label>
      <input
        ref={firstFormElementRef}
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
