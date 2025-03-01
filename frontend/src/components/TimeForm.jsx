import "./Styles/timeForm.css";
import React, { useState } from "react";
import Calendar from "react-calendar";

export default function TimeForm({ state, setState, text, defaultText, id }) {
  const [date, setDate] = useState(new Date()); // Track the selected date
  const [showCalendar, setShowCalendar] = useState(false); // Track the visibility of the calendar

  const handleDateChange = (newDate) => {
    setDate(newDate); // Update the selected date
    setState(newDate); // Optionally update the parent state
    setShowCalendar(false); // Hide the calendar after selecting a date
  };

  const formatDate = (date) => {
    // Format the date to display in the input field
    return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
  };

  const handleInputFocus = () => {
    setShowCalendar(true); // Show the calendar when the input field is focused
  };

  const handleInputBlur = (e) => {
    // Prevent calendar from disappearing if clicked inside the calendar
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setShowCalendar(false);
    }
  };

  return (
    <div className="input-group">
      <label htmlFor="payment">{text}</label>
      <input
        type="text"
        id={id}
        name={id}
        placeholder={defaultText}
        value={formatDate(date)} // Display formatted date in the input field
        onChange={(e) => setState(e.target.value)} // Allow manual change if necessary
        onFocus={handleInputFocus} // Show calendar when input is clicked
        onBlur={handleInputBlur} // Hide calendar when input loses focus
      />
      
      {showCalendar && (
        <div className="calendar-container">
          <Calendar onChange={handleDateChange} value={date} />
        </div>
      )}
    </div>
  );
}
