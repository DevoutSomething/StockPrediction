import { useContext } from "react";
import { GlobalContext } from "../context";
import Form from "../components/Form";
import TimeForm from "../components/TimeForm";
import "./Styles/formContainer.css";

export default function FormContainer() {
  const { payment, setPayment, time, setTime, profit, setProfit } =
    useContext(GlobalContext);

  // Check if all fields are set
  const allFieldsSet = payment && time && profit;

  return (
    <div className="page-container">
      <h1 className="form-heading">Investment Calculator</h1>
      <div className={`form-container-wrapper ${allFieldsSet ? "filled" : ""}`}>
        <p className="form-description">Your stock journey starts here.</p>
        <div className="form-container">
          <Form
            isFirst={true}
            state={payment}
            setState={setPayment}
            text="Investment Amount"
            id="payment"
            defaultText="Enter Investment"
          />
          <Form
            state={profit}
            setState={setProfit}
            text="Desired Profit"
            id="profit"
            defaultText="Enter Desired Profit"
          />
          <TimeForm
            state={time}
            setState={setTime}
            text="Time Frame"
            id="time"
            defaultText="Enter When to Sell"
          />
        </div>
      </div>
    </div>
  );
}