import "./Styles/form.css";

export default function Form({ state, setState, text, defaultText, id }) {
  return (
    <div className="input-group">
      <label htmlFor="payment">{text}</label>
      <input
        type="number"
        id={id}
        name={id}
        placeholder={defaultText}
        value={state}
        onChange={(e) => setState(e.target.value)}
      />
    </div>
  );
}
