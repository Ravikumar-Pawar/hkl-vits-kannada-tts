async function generate() {


    const textInput =
        document.getElementById("text");


    const text =
        textInput.value.trim();


    if (!text) {

        alert("Please enter Kannada text");

        return;
    }


    const button =
        document.getElementById("generate-btn");


    if (button) {

        button.disabled = true;

        button.innerText = "Generating...";

    }


    try {


        const response = await fetch(
            "/api/tts",
            {

                method: "POST",

                headers: {

                    "Content-Type": "application/json"

                },


                body: JSON.stringify({

                    text: text

                })

            }
        );


        if (!response.ok) {


            const error =
                await response.json();


            alert(
                error.detail || "TTS generation failed"
            );


            return;
        }


        const audioBlob =
            await response.blob();


        const audioURL =
            URL.createObjectURL(
                audioBlob
            );


        const player =
            document.getElementById(
                "player"
            );


        player.src = audioURL;


        player.load();


        await player.play();


    } catch (error) {


        console.error(
            "TTS Error:",
            error
        );


        alert(
            "Failed to generate speech"
        );


    } finally {


        if (button) {

            button.disabled = false;

            button.innerText = "Generate";

        }

    }

}