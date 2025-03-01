import { createContext, useState } from "react";

export const GlobalContext = createContext(null);

export default function GlobalState({ children }) {
  const [payment, setPayment] = useState(0);
  const [time, setTime] = useState(0);
  const [profit, setProfit] = useState(0);
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
      {" "}
      {children}
    </GlobalContext.Provider>
  );
}
