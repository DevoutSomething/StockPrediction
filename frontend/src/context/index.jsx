import { createContext, useState } from "react";

export const GlobalContext = createContext(null);

export default function GlobalState({ children }) {
  const [payment, setPayment] = useState("");
  const [time, setTime] = useState("");
  const [profit, setProfit] = useState("");
  return (
    <GlobalContext.Provider
      value={{
        payment,
        setPayment,
        time,
        setTime,
        profit,
        setProfit,
      }}
    >
      {children}
    </GlobalContext.Provider>
  );
}
