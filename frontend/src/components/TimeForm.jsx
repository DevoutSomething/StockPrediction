import React, { useState, useRef } from "react";
import "react-calendar/dist/Calendar.css";
import Calendar from "react-calendar";
import "./Styles/timeForm.css";

export default function TimeForm({ state, setState, text, defaultText, id }) {
  const [date, setDate] = useState(new Date());
  const [showCalendar, setShowCalendar] = useState(false);
  const [error, setError] = useState("");
  const calendarRef = useRef(null);

  const handleDateChange = (newDate) => {
    const today = new Date();
    today.setHours(0, 0, 0, 0); // Reset time to only compare dates

    if (newDate < today) {
      setError("Please select a future date.");
      return;
    }

    setError("");
    setDate(newDate);
    setState(newDate);
    setShowCalendar(false); // Close calendar after valid selection
  };

  const formatDate = (date) => {
    return `${date.getFullYear()}-${(date.getMonth() + 1)
      .toString()
      .padStart(2, "0")}-${date.getDate().toString().padStart(2, "0")}`;
  };

  const handleInputFocus = () => {
    setShowCalendar(true);
  };

  const handleClickOutside = (e) => {
    if (calendarRef.current && !calendarRef.current.contains(e.target)) {
      setShowCalendar(false);
    }
  };

  React.useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  // Apply the 'valid' class if a date is selected
  const inputClass = state ? "valid" : "";

  return (
    <div className="input-group">
      <label htmlFor={id}>{text}</label>
      <input
        type="text"
        id={id}
        name={id}
        placeholder={defaultText}
        value={!state ? state : formatDate(state)}
        onFocus={handleInputFocus}
        className={inputClass} // Apply the 'valid' class if a date is selected
        readOnly
      />
      {error && <div className="error">{error}</div>}
      {showCalendar && (
        <div className="calendar-container" ref={calendarRef}>
          <Calendar onChange={handleDateChange} value={date} />
        </div>
      )}
    </div>
  );
}
