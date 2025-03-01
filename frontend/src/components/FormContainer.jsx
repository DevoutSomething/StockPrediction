import { useContext } from "react";
import { GlobalContext } from "../context";
import Form from "../components/Form";
import TimeForm from "../components/TimeForm";
import "./Styles/formContainer.css";

export default function FormContainer() {
  const { payment, setPayment, time, setTime, profit, setProfit } =
    useContext(GlobalContext);

  return (
    <div className="form-container">
      <Form
        isFirst={true}
        state={payment}
        setState={setPayment}
        text="Payment"
        id="payment"
        defaultText="Enter Payment"
      />
      <Form
        state={profit}
        setState={setProfit}
        text="Profit"
        id="profit"
        defaultText="Enter Profit"
      />
      <TimeForm
        state={time}
        setState={setTime}
        text="Time"
        id="time"
        defaultText="Enter Time"
      />
    </div>
  );
}
