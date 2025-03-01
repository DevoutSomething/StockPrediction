import React, { useState, useRef } from "react";
import "react-calendar/dist/Calendar.css";
import Calendar from "react-calendar";

export default function TimeForm({ state, setState, text, defaultText, id }) {
  const [date, setDate] = useState(new Date());
  const [showCalendar, setShowCalendar] = useState(false);
  const calendarRef = useRef(null);

  const handleDateChange = (newDate) => {
    setDate(newDate);
    setState(newDate);
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

  return (
    <div className="input-group">
      <label htmlFor={id}>{text}</label>
      <input
        type="text"
        id={id}
        name={id}
        placeholder={defaultText}
        value={!state ? state : formatDate(state)}
        onChange={(e) => setState(e.target.value)}
        onFocus={handleInputFocus}
      />
      {showCalendar && (
        <div className="calendar-container" ref={calendarRef}>
          <Calendar onChange={handleDateChange} value={date} />
        </div>
      )}
    </div>
  );
}
