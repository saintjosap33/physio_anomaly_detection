export async function predictVitals(vitals: any) {
  try {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(vitals),
    });

    const data = await res.json();
    return data;

  } catch (err) {
    console.error("API ERROR:", err);
    return null;
  }
}