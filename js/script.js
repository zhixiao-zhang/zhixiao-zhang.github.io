document.addEventListener("DOMContentLoaded", function () {
  const menuToggle = document.getElementById("menu-toggle");
  const menu = document.getElementById("menu");
  const header = document.querySelector(".header");
  const menuItems = menu.querySelectorAll("li a");

  menuToggle.addEventListener("click", function () {
    // 切换菜单的隐藏/显示
    menu.classList.toggle("hidden");
    menu.classList.toggle("visible");

    if (menu.classList.contains("visible")) {
      header.classList.add("expanded");
    } else {
      header.classList.remove("expanded");
    }
  });

  menuItems.forEach(function (menuItem) {
    menuItem.addEventListener("click", function () {
      menu.classList.toggle("visible");
      header.classList.remove("expanded");
    });
  });
});

  function copyToClipboard(event) {
    const button = event.currentTarget;
    const targetId = button.getAttribute('data-target');
    const textElement = document.getElementById(targetId);

    if (textElement) {
      // Copy text to clipboard
      navigator.clipboard.writeText(textElement.textContent)
        .then(() => {
          // Change button text to a checkmark ✔
          const originalText = button.textContent;
          button.textContent = "✔";
          button.disabled = true; // Temporarily disable the button
          
          // Restore original text after 2 seconds
          setTimeout(() => {
            button.textContent = originalText;
            button.disabled = false; // Re-enable the button
          }, 2000);
        })
        .catch(err => {
          console.error('Failed to copy text: ', err);
        });
    }
  }
