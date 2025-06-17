document.addEventListener("DOMContentLoaded", function () {
    const clearBtn = document.getElementById("clear-btn");
    const textArea = document.querySelector("textarea");

    clearBtn.addEventListener("click", () => {
        textArea.value = "";
        window.location.href = "/";
    });
});
