import random
from ..config import Config

class PromptGenerator:
    """Generates prompts optimized for the Ghibli-Diffusion model"""
    
    def __init__(self):
        """Initialize the prompt generator"""
        self.config = Config()
        self.prompt_config = self.config.prompts
        self.ghibli_style = self.prompt_config['styles']['studio_ghibli']
        self.negative_prompt = self.prompt_config['negative_prompt']
        
        # Ghibli-specific prompts and elements
        self.ghibli_characters = [
            "Totoro", "Kodama forest spirits", "No-Face", "Jiji the cat", "Kiki",
            "Howl", "Calcifer", "Ponyo", "Haku the dragon", "soot sprites",
            "Chihiro", "Nausicaä", "San", "Princess Mononoke", "Ashitaka"
        ]
        
        self.ghibli_locations = [
            "a peaceful mountain village", "a magical forest", "a seaside town",
            "a floating castle", "a spirit bathhouse", "a witch's house", 
            "an abandoned amusement park", "a flying castle", "a magical garden",
            "a mountain valley", "a cozy cottage", "a hidden forest grove"
        ]
        
        self.ghibli_elements = [
            "river spirits", "magical creatures", "gentle giants", "flying machines",
            "fluffy clouds", "ancient magic", "soft rainfall", "cherry blossoms",
            "floating lanterns", "glowing orbs", "magical transformation", 
            "windswept grass", "flowing rivers", "magical amulets", "paper charms"
        ]
    
    def generate_prompt(self, stock_data=None):
        """Generate a Studio Ghibli style prompt based on stock data"""
        # Determine the prompt elements based on stock performance
        if stock_data:
            # Check stock movement direction
            is_rising = stock_data['price_change'] > 0
            is_falling = stock_data['price_change'] < 0
            is_stable = not is_rising and not is_falling
            
            # Get magnitude of movement (0-10 scale)
            if is_rising or is_falling:
                magnitude = min(10, abs(stock_data['price_change_pct']) / 2.0)
            else:
                magnitude = 0
                
            # Select base prompt structure based on stock movement
            if is_rising:
                base_prompt = self._get_rising_prompt(magnitude)
            elif is_falling:
                base_prompt = self._get_falling_prompt(magnitude)
            else:
                base_prompt = self._get_stable_prompt()
        else:
            # Random prompt if no stock data
            base_prompt = random.choice([
                "peaceful Ghibli landscape",
                "a vibrant Ghibli town",
                "magical Ghibli forest",
                "a cozy Ghibli cottage"
            ])
        
        # Always add Ghibli style suffix for better results with the model
        ghibli_suffix = "Studio Ghibli style, Miyazaki style, fantasy art"
        
        # Add random Ghibli elements
        elements = []
        elements.append(random.choice(self.ghibli_locations))
        
        # 40% chance to add a character
        if random.random() < 0.4:
            elements.append(random.choice(self.ghibli_characters))
            
        # 70% chance to add a magical element
        if random.random() < 0.7:
            elements.append(random.choice(self.ghibli_elements))
            
        # Create final prompt
        prompt_elements = [base_prompt] + elements + [ghibli_suffix]
        prompt = ", ".join(prompt_elements)
        
        return prompt
    
    def _get_rising_prompt(self, magnitude):
        """Generate prompt for rising stock price
        
        Args:
            magnitude: 0-10 scale indicating strength of movement
        
        Returns:
            Base prompt string
        """
        if magnitude < 3:
            # Small rise
            return random.choice([
                "gentle sunrise over",
                "spring blooms in",
                "soft morning light in",
                "budding flowers in",
            ])
        elif magnitude < 7:
            # Medium rise
            return random.choice([
                "soaring airship above",
                "floating lanterns rising from",
                "magical transformation in",
                "flying castle above"
            ])
        else:
            # Large rise
            return random.choice([
                "spectacular dragon flying over",
                "magical explosion of light in",
                "triumphant heroes overlooking",
                "majestic mountain peaks with"
            ])
    
    def _get_falling_prompt(self, magnitude):
        """Generate prompt for falling stock price
        
        Args:
            magnitude: 0-10 scale indicating strength of movement
        
        Returns:
            Base prompt string
        """
        if magnitude < 3:
            # Small fall
            return random.choice([
                "gentle rainfall over",
                "autumn leaves falling in",
                "peaceful dusk in",
                "light fog rolling through"
            ])
        elif magnitude < 7:
            # Medium fall
            return random.choice([
                "stormy weather approaching",
                "character looking down from",
                "floating down a river through",
                "descending staircase in"
            ])
        else:
            # Large fall
            return random.choice([
                "dramatic waterfall cascading down",
                "character falling through clouds above",
                "abandoned ruins in",
                "mysterious deep valley with"
            ])
    
    def _get_stable_prompt(self):
        """Generate prompt for stable stock price
        
        Returns:
            Base prompt string
        """
        return random.choice([
            "tranquil scene in",
            "peaceful day in",
            "balanced harmony in",
            "quiet moment in",
            "still waters reflecting"
        ])
    
    def get_negative_prompt(self):
        """Return the negative prompt"""
        return self.negative_prompt
