document.addEventListener('DOMContentLoaded', function() {
    // Password visibility toggle
    const setupPasswordToggles = function() {
        // Handle password toggles on any page
        const passwordToggles = [
            { toggleId: 'togglePassword', inputId: 'password' },
            { toggleId: 'toggleConfirmPassword', inputId: 'confirm_password' }
        ];
        
        passwordToggles.forEach(config => {
            const toggleBtn = document.getElementById(config.toggleId);
            const passwordInput = document.getElementById(config.inputId);
            
            if (toggleBtn && passwordInput) {
                toggleBtn.onclick = function() {
                    // Check if password is currently hidden
                    const isPasswordHidden = passwordInput.type === 'password';
                    
                    // Toggle password visibility
                    passwordInput.type = isPasswordHidden ? 'text' : 'password';
                    
                    // Update icon
                    const icon = toggleBtn.querySelector('i');
                    if (icon) {
                        icon.className = isPasswordHidden ? 'fas fa-eye' : 'fas fa-eye-slash';
                    }
                };
            }
        });
    };
    
    setupPasswordToggles();
    
    // Handle authentication redirects for protected pages
    const handleProtectedNavLinks = function() {
        // Protected routes that require login
        const protectedRoutes = [
            { path: '/predict', name: 'Prediction' },
            { path: '/analysis', name: 'Analysis' },
            { path: '/history', name: 'History' }
        ];
        
        // Check if user is logged in
        const isLoggedIn = document.body.classList.contains('logged-in') || 
                          document.querySelector('.navbar .dropdown-toggle') !== null;
        
        // Set up listeners for nav links to protected routes
        protectedRoutes.forEach(route => {
            const navLinks = document.querySelectorAll(`a[href="${route.path}"]`);
            
            navLinks.forEach(link => {
                // Only handle clicks if not logged in
                if (!isLoggedIn) {
                    link.addEventListener('click', function(e) {
                        e.preventDefault();
                        
                        // Show login modal if it exists
                        const loginModal = document.getElementById('loginRequiredModal');
                        if (loginModal) {
                            const modalBody = loginModal.querySelector('.modal-body');
                            if (modalBody) {
                                modalBody.innerHTML = `You need to login to access the ${route.name} page.`;
                            }
                            
                            // Get the login button
                            const loginButton = loginModal.querySelector('.login-redirect-btn');
                            if (loginButton) {
                                loginButton.href = `/auth/login?next=${route.path}`;
                            }
                            
                            const modal = new bootstrap.Modal(loginModal);
                            modal.show();
                        } else {
                            // If no modal, redirect directly to login
                            window.location.href = `/auth/login?next=${route.path}`;
                        }
                    });
                }
            });
        });
    };
    
    // Initialize protected routes handling
    handleProtectedNavLinks();
    
    // Add smooth scroll animation for internal links
    const smoothScrollLinks = document.querySelectorAll('a[href^="#"]');
    smoothScrollLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Animation on scroll
    const animateElements = document.querySelectorAll('.animate-on-scroll');
    
    function checkScroll() {
        const windowHeight = window.innerHeight;
        const scrollY = window.scrollY || window.pageYOffset;
        
        animateElements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top + scrollY;
            
            if (scrollY > elementPosition - windowHeight + 100) {
                element.classList.add('show');
            }
        });
    }
    
    window.addEventListener('scroll', checkScroll);
    checkScroll(); // Check on page load

    // Card hover effects
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.1)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });
    
    // Form validation
    const form = document.getElementById('prediction-form');
    
    if (form) {
        // Toggle switches for medical history
        const toggles = [
            { toggleId: 'asthma-toggle', valueId: 'asthma-value', indicatorClass: 'asthma-indicator' },
            { toggleId: 'smoking-toggle', valueId: 'smoking-value', indicatorClass: 'smoking-indicator' },
            { toggleId: 'pad-toggle', valueId: 'pad-value', indicatorClass: 'pad-indicator' },
            { toggleId: 'mi-toggle', valueId: 'mi-value', indicatorClass: 'mi-indicator' },
            { toggleId: 'diabetes-toggle', valueId: 'diabetes-value', indicatorClass: 'diabetes-indicator' }
        ];

        toggles.forEach(item => {
            const toggle = document.getElementById(item.toggleId);
            const valueInput = document.getElementById(item.valueId);
            const indicator = document.querySelector(`.${item.indicatorClass}`);

            if (toggle && valueInput && indicator) {
                toggle.addEventListener('change', function() {
                    if (this.checked) {
                        valueInput.value = 'yes';
                        indicator.style.width = '100%';
                    } else {
                        valueInput.value = 'no';
                        indicator.style.width = '0%';
                    }
                });
            }
        });

        form.addEventListener('submit', function(event) {
            // Get all the numeric inputs
            const fvc = parseFloat(document.getElementById('fvc').value);
            const fev1 = parseFloat(document.getElementById('fev1').value);
            const age = parseInt(document.getElementById('age').value);
            
            // Validate FVC
            if (isNaN(fvc) || fvc <= 0 || fvc > 10) {
                event.preventDefault();
                showError('fvc', 'FVC value must be between 0 and 10 liters');
                return;
            } else {
                clearError('fvc');
            }
            
            // Validate FEV1
            if (isNaN(fev1) || fev1 <= 0 || fev1 > 10) {
                event.preventDefault();
                showError('fev1', 'FEV1 value must be between 0 and 10 liters');
                return;
            } else {
                clearError('fev1');
            }
            
            // Validate Age
            if (isNaN(age) || age < 20 || age > 130) {
                event.preventDefault();
                showError('age', 'Age must be between 20 and 130 years');
                return;
            } else {
                clearError('age');
            }
            
            // Add loading indicator
            const submitBtn = document.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Processing...';
                submitBtn.disabled = true;
            }
        });
    }
    
    // Function to show error message
    function showError(inputId, message) {
        const input = document.getElementById(inputId);
        if (input) {
            input.classList.add('is-invalid');
            
            // Create or update error message
            let errorElement = document.getElementById(`${inputId}-error`);
            if (!errorElement) {
                errorElement = document.createElement('div');
                errorElement.className = 'invalid-feedback';
                errorElement.id = `${inputId}-error`;
                input.parentNode.appendChild(errorElement);
            }
            errorElement.textContent = message;
        }
    }
    
    // Function to clear error message
    function clearError(inputId) {
        const input = document.getElementById(inputId);
        if (input) {
            input.classList.remove('is-invalid');
            
            // Remove error message
            const errorElement = document.getElementById(`${inputId}-error`);
            if (errorElement) {
                errorElement.remove();
            }
        }
    }
    
    // Add visual feedback when form fields are modified
    const formControls = document.querySelectorAll('.form-control');
    formControls.forEach(function(control) {
        control.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        control.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });
    
    // Initialize risk circles on result page
    const riskCircles = document.querySelectorAll('.risk-circle');
    riskCircles.forEach(circle => {
        const percentage = circle.style.getPropertyValue('--percentage') || '0%';
        const percentageValue = parseFloat(percentage);
        
        setTimeout(() => {
            circle.style.background = circle.classList.contains('risk-circle-high') 
                ? `conic-gradient(var(--danger-color) 0% ${percentageValue}%, #f1f1f1 ${percentageValue}% 100%)`
                : `conic-gradient(var(--success-color) 0% ${percentageValue}%, #f1f1f1 ${percentageValue}% 100%)`;
        }, 100);
    });
}); 
// patient analysiis page
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the risk circle
    const riskCircle1 = document.getElementById('riskCircle');
    if (riskCircle1) {
        const percentage = riskCircle1.getAttribute('data-percentage');
        const percentageValue = parseFloat(percentage);
        
        // Determine if it's high risk or low risk
        const isHighRisk = document.querySelector('h3.text-danger') !== null;
        
        // Set the background with conic gradient
        setTimeout(() => {
            if (isHighRisk) {
                riskCircle.style.background = `conic-gradient(var(--danger-color) 0% ${percentageValue}%, #f1f1f1 ${percentageValue}% 100%)`;
            } else {
                riskCircle.style.background = `conic-gradient(var(--success-color) 0% ${percentageValue}%, #f1f1f1 ${percentageValue}% 100%)`;
            }
        }, 100);
    }
});