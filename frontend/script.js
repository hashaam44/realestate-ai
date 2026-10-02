const predictionForm = document.getElementById("predictionForm");

const predictButton = document.getElementById("predictButton");
const buttonText = document.getElementById("buttonText");

const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent = document.getElementById("resultContent");
const predictedPrice = document.getElementById("predictedPrice");


predictionForm.addEventListener("submit", async function (event) {

    event.preventDefault();

    predictButton.disabled = true;
    buttonText.textContent = "Predicting...";


    const propertyData = {
        bed: Number(document.getElementById("bed").value),
        bath: Number(document.getElementById("bath").value),
        acre_lot: Number(document.getElementById("acre_lot").value),
        house_size: Number(document.getElementById("house_size").value),
        prev_sold_year: Number(
            document.getElementById("prev_sold_year").value
        ),
        status: document.getElementById("status").value,
        state: document.getElementById("state").value.trim()
    };


    try {

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(propertyData)
            }
        );


        const result = await response.json();


        if (!response.ok) {
            throw new Error(
                result.detail || "Prediction failed."
            );
        }


        const formattedPrice = new Intl.NumberFormat(
            "en-US",
            {
                style: "currency",
                currency: "USD",
                maximumFractionDigits: 0
            }
        ).format(result.predicted_price);


        predictedPrice.textContent = formattedPrice;

        resultPlaceholder.style.display = "none";
        resultContent.style.display = "block";


    } catch (error) {

        console.error("Prediction error:", error);

        resultPlaceholder.style.display = "block";
        resultContent.style.display = "none";

        alert(error.message);

    } finally {

        predictButton.disabled = false;
        buttonText.textContent = "Predict Property Price";
    }
});