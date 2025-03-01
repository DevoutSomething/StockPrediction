import React, { useRef, useEffect, useState } from "react";
import "./Styles/form.css";

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
  const [rawValue, setRawValue] = useState(state);

  const handleInputChange = (e) => {
    const value = e.target.value;

    if (/^\d*\.?\d*$/.test(value)) {
      setRawValue(value);
    }
  };

  const handleBlur = () => {
    if (rawValue) {
      const numericValue = parseFloat(rawValue);
      if (!isNaN(numericValue)) {
        const formattedValue = `$${numericValue.toLocaleString("en-US", {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}`;
        setState(numericValue.toString()); // Update parent state with numeric value
        setRawValue(formattedValue); // Update local state with formatted value
      }
    } else {
      setState(rawValue); // Update parent state with raw value
    }
  };

  const handleFocus = () => {
    if (rawValue) {
      const numericValue = rawValue.replace(/[^0-9.]/g, "");
      setRawValue(numericValue);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" || e.keyCode === 13) {
      e.target.blur();
    }
  };

  const firstFormElementRef = useRef(null);

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
        type="text" // Use "text" type to allow flexible input
        id={id}
        name={id}
        placeholder={defaultText}
        value={rawValue}
        onChange={handleInputChange}
        onBlur={handleBlur}
        onFocus={handleFocus}
        onKeyDown={handleKeyDown} // Add keydown handler
        className={inputClass}
      />
    </div>
  );
}
