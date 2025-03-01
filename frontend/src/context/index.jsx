import { createContext, useState } from "react";

export const GlobalContext = createContext(null);

export default function GlobalState({ children }) {
  const [payment, setPayment] = useState(null);
  const [time, setTime] = useState(null);
  const [profit, setProfit] = useState(null);
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
