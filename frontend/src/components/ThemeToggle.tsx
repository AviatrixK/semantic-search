import { useTheme } from '../hooks/useTheme'
import { themeText } from '../lib/theme'
import Icon from './Icon'
import styles from './ThemeToggle.module.css'

/** Light <-> dark. The icon shows what a click switches TO; the label spells it out. `menu` makes it a row of the account menu. */
export default function ThemeToggle({ menu = false }: { menu?: boolean }) {
  const { choice, cycle } = useTheme()
  const text = themeText(choice)
  return (
    <button type="button" className={menu ? styles.row : styles.toggle} onClick={cycle} aria-label={text.label} title={text.label} role={menu ? 'menuitem' : undefined}>
      <Icon name={choice === 'light' ? 'moon' : 'sun'} size={menu ? 22 : 20} />
      {menu && <span>Appearance: {text.current}</span>}
    </button>
  )
}
