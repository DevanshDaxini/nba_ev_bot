"""
sports_ev_bot - Multi-Sport Entry Point

Interactive menu system for NBA, CBB, and upcoming sports betting analysis.

Features:
    - Sport selection menu
    - Separate workflows for each sport
    - Shared core functionality (FanDuel, PrizePicks APIs)
    
Sports Supported:
    - NBA (Professional Basketball) - ACTIVE
    - CBB (College Basketball) - ACTIVE
    - WNBA (Women's Basketball) - COMING SOON
    - MLB (Major League Baseball) - COMING SOON
    - NFL (Football) - COMING SOON

Usage:
    $ python main.py
    
Then select your sport and follow the prompts.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main entry point - Sport selection menu.
    
    Workflow:
        1. Display sport menu
        2. User selects sport
        3. Launch sport-specific CLI
        4. Return to sport menu (or exit)
    """
    while True:
        # Clear screen for clean display
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*60)
        print("  " + "🏀"*8 + "  SPORTS ANALYTICS HUB  " + "🏀"*8)
        print("="*60)
        
        print("\n🎯 PROFESSIONAL BETTING ANALYSIS PLATFORM")
        print("   AI-Powered Predictions • Multi-Sport Support • Real-Time Odds\n")
        
        print("="*60)
        print("SELECT SPORT")
        print("="*60)
        
        # Active Sports
        print("\n🟢 ACTIVE")
        
        print("\n1. 🏀 NBA")
        print("   Professional Basketball")
        print("   ⭐ Elite Models: PTS (89%), FGM (88%), PA (87%)")
        print("   📊 Active Models: 13 stats")
        print("   💰 PrizePicks Breakeven: 54.1%")
        
        print("\n" + "="*60)
        print("\n0. 🚪 Exit")
        
        print("\n" + "="*60)
        
        choice = input("\nSelect Sport (1 or 0 to exit): ").strip()
        
        # ================================================================
        # ACTIVE SPORTS
        # ================================================================

        if choice == '1':
            # Launch NBA
            try:
                from src.cli.nba_cli import main_menu as nba_menu
                nba_menu()
            except ImportError as e:
                print(f"\n❌ Error loading NBA module: {e}")
                print("   Make sure src/cli/nba_cli.py exists")
                input("\nPress Enter to continue...")
            except Exception as e:
                print(f"\n❌ NBA module error: {e}")
                input("\nPress Enter to continue...")
        
        # ================================================================
        # EXIT
        # ================================================================
        
        elif choice == '0':
            # Exit
            print("\n" + "="*60)
            print("  👋 GOODBYE!")
            print("="*60)
            print("\n📊 Session Summary:")
            print("   Thanks for using Sports Analytics Hub")
            print("   Good luck with your bets! 🎯")
            print("\n💡 Tips:")
            print("   - Stick to ELITE tier models (highest accuracy)")
            print("   - Check injury reports before betting")
            print("   - Manage bankroll wisely (never bet more than 3%)")
            # print("\n🔮 Coming Soon: WNBA (May 2026), MLB (Spring 2026), NFL (Summer 2026)")
            print("\n" + "="*60 + "\n")
            break
        
        else:
            print("\n❌ Invalid selection. Please choose 1 or 0.")
            input("Press Enter to try again...")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("\nPlease report this issue if it persists.")