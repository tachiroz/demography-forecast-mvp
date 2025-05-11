import { useState } from 'react';

interface Props {
  selected: string;                  // текущий выбор в селекте
  onSelect: (m: string) => void;     // меняем выбор
  onTrained: () => void;             // сообщаем, что обучение завершилось
}

export default function ModelForm({ selected, onSelect, onTrained }: Props) {
  const [loading, setLoading] = useState(false);

  const train = async () => {
    setLoading(true);
    await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selected, params: {} }),
    });
    setLoading(false);
    onTrained();                     // сигнал «готово»
  };

  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ marginRight: 8 }}>Model:</label>
      <select value={selected} onChange={e => onSelect(e.target.value)}>
        <option value="sarimax">SARIMAX (Births)</option>
        <option value="sarimax_pop">SARIMAX-POP (Population)</option>
        <option value="prophet">Prophet</option>
        <option value="xgb">XGBoost</option>
        <option value="cat">CatBoost</option>
      </select>

      <button style={{ marginLeft: 12 }} onClick={train} disabled={loading}>
        {loading ? 'Training…' : 'Train'}
      </button>
    </div>
  );
}
